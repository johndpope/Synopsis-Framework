//
//  OpenCVAnalyzerPlugin.m
//  MetadataTranscoderTestHarness
//
//  Created by vade on 4/3/15.
//  Copyright (c) 2015 Synopsis. All rights reserved.
//

#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreGraphics/CoreGraphics.h>

#import "StandardAnalyzerPlugin.h"

#import "StandardAnalyzerDefines.h"

// CPU Modules
#import "AverageColor.h"
#import "DominantColorModule.h"
#import "HistogramModule.h"
#import "MotionModule.h"
#import "PerceptualHashModule.h"
#import "TrackerModule.h"
#import "SaliencyModule.h"
//#import "TensorflowFeatureModule.h"

// GPU Module
#import "GPUHistogramModule.h"
#import "GPUVisionMobileNet.h"
#import "GPUMPSMobileNet.h"

@interface StandardAnalyzerPlugin ()
{
}

#pragma mark - Plugin Protocol Requirements

@property (atomic, readwrite, strong) NSString* pluginName;
@property (atomic, readwrite, strong) NSString* pluginIdentifier;
@property (atomic, readwrite, strong) NSArray* pluginAuthors;
@property (atomic, readwrite, strong) NSString* pluginDescription;
@property (atomic, readwrite, assign) NSUInteger pluginAPIVersionMajor;
@property (atomic, readwrite, assign) NSUInteger pluginAPIVersionMinor;
@property (atomic, readwrite, assign) NSUInteger pluginVersionMajor;
@property (atomic, readwrite, assign) NSUInteger pluginVersionMinor;
@property (atomic, readwrite, strong) NSString* pluginMediaType;
@property (atomic, readwrite, strong) dispatch_queue_t serialDictionaryQueue;

// We create a serial operation queue for every module
// This allows us to run modules in parallel, create dependency chains
// But also gurantees a single module is never running in parallel.

@property (atomic, readwrite, strong) NSArray<NSOperationQueue*>* moduleOperationQueues;
@property (atomic, readwrite, strong) NSMutableDictionary* lastModuleOperation;

#pragma mark - Analyzer Modules

@property (atomic, readwrite, strong) NSArray* cpuModuleClasses;
@property (atomic, readwrite, strong) NSMutableArray<CPUModule*>* cpuModules;

@property (atomic, readwrite, strong) NSArray* gpuModuleClasses;
@property (atomic, readwrite, strong) NSMutableArray<GPUModule*>* gpuModules;

#pragma mark - Ingest

@property (atomic, readwrite, strong) SynopsisVideoFrameCache* lastFrameCache;
@property (readwrite, strong) NSArray<SynopsisVideoFormatSpecifier*>*pluginFormatSpecfiers;

@property (readwrite, strong) id<MTLDevice> device;

@property (readwrite, assign) BOOL didLazyInitModulesAlready;


@end

@implementation StandardAnalyzerPlugin

- (id) init
{
    self = [super init];
    if(self)
    {
        self.pluginName = @"Standard Analyzer";
        self.pluginIdentifier = kSynopsisStandardMetadataDictKey;
        self.pluginAuthors = @[@"Anton Marini"];
        self.pluginDescription = @"Standard Analyzer, providing Color, Features, Content Tagging, Histogram, Motion";
        self.pluginAPIVersionMajor = 0;
        self.pluginAPIVersionMinor = 1;
        self.pluginVersionMajor = 0;
        self.pluginVersionMinor = 1;
        self.pluginMediaType = AVMediaTypeVideo;

        self.cpuModules = [NSMutableArray new];
        self.gpuModules = [NSMutableArray new];

        self.cpuModuleClasses  = @[// AVG Color is useless and just an example module
//                                [AverageColor className],
                                   NSStringFromClass([DominantColorModule class]),
                                   NSStringFromClass([HistogramModule class]),
//                                   [MotionModule className],
//                                   [TensorflowFeatureModule className],
//                                   [TrackerModule className],
//                                   [SaliencyModule className],
                                   ];

        // Disable CPU for now:
//        self.cpuModuleClasses = @[];
        
        self.gpuModuleClasses  = @[
//                                  NSStringFromClass([GPUHistogramModule class]),
                                  NSStringFromClass([GPUVisionMobileNet class]),
//                                  [GPUMPSMobileNet className],
                                   ];
        
        NSMutableArray<SynopsisVideoFormatSpecifier*>*requiredSpecifiers = [NSMutableArray new];
        for(NSString* moduleClass in self.cpuModuleClasses)
        {
            Class module = NSClassFromString(moduleClass);
            SynopsisVideoFormatSpecifier* format = [[SynopsisVideoFormatSpecifier alloc] initWithFormat:[module requiredVideoFormat] backing:[module requiredVideoBacking]];
            [requiredSpecifiers addObject:format];
        }
       
        for(NSString* moduleClass in self.gpuModuleClasses)
        {
            Class module = NSClassFromString(moduleClass);
            SynopsisVideoFormatSpecifier* format = [[SynopsisVideoFormatSpecifier alloc] initWithFormat:[module requiredVideoFormat] backing:[module requiredVideoBacking]];
            [requiredSpecifiers addObject:format];
        }
        
        self.pluginFormatSpecfiers = requiredSpecifiers;
        
        NSMutableArray<NSOperationQueue*>* moduleQueues = [NSMutableArray new];
        
        [self.cpuModuleClasses enumerateObjectsUsingBlock:^(CPUModule * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {

            NSOperationQueue* moduleQueue = [[NSOperationQueue alloc] init];
            moduleQueue.maxConcurrentOperationCount = 1;
            
            [moduleQueues addObject:moduleQueue];
        }];
        
        self.moduleOperationQueues = [moduleQueues copy];
        
        self.serialDictionaryQueue = dispatch_queue_create("module_queue", DISPATCH_QUEUE_CONCURRENT_WITH_AUTORELEASE_POOL);
        
        self.didLazyInitModulesAlready = NO;
        
    }
    
    return self;
}

- (void) beginMetadataAnalysisSessionWithQuality:(SynopsisAnalysisQualityHint)qualityHint device:(id<MTLDevice>)device;
{
//    dispatch_async(dispatch_get_main_queue(), ^{
//        cv::namedWindow("OpenCV Debug", CV_WINDOW_NORMAL);
//    });
    
    if(!self.didLazyInitModulesAlready)
    {
        self.device = device;
        
        for(NSString* classString in self.cpuModuleClasses)
        {
            Class moduleClass = NSClassFromString(classString);
            
            CPUModule* module = [(CPUModule*)[moduleClass alloc] initWithQualityHint:qualityHint];
            
            if(module != nil)
            {
                [self.cpuModules addObject:module];
                
                if(self.verboseLog)
                    self.verboseLog([@"Loaded Module: " stringByAppendingString:classString]);
            }
        }
        
        for(NSString* classString in self.gpuModuleClasses)
        {
            Class moduleClass = NSClassFromString(classString);
            
            GPUModule* module = [(GPUModule*)[moduleClass alloc] initWithQualityHint:qualityHint device:self.device];
            
            if(module != nil)
            {
                [self.gpuModules addObject:module];
                
                if(self.verboseLog)
                    self.verboseLog([@"Loaded Module: " stringByAppendingString:classString]);
            }
        }
        
        self.didLazyInitModulesAlready = YES;
    }
    
    [self.cpuModules enumerateObjectsUsingBlock:^(CPUModule * _Nonnull module, NSUInteger idx, BOOL * _Nonnull stop) {
        [module beginAndClearCachedResults];
    }];

    [self.gpuModules enumerateObjectsUsingBlock:^(GPUModule * _Nonnull module, NSUInteger idx, BOOL * _Nonnull stop) {
        [module beginAndClearCachedResults];
    }];
}

- (void) analyzeFrameCache:(SynopsisVideoFrameCache*)frameCache commandBuffer:(id<MTLCommandBuffer>)frameCommandBuffer completionHandler:(SynopsisAnalyzerPluginFrameAnalyzedCompleteCallback)completionHandler
{
//    static NSUInteger frameSubmit = 0;
//    static NSUInteger frameComplete = 0;

    NSMutableDictionary* dictionary = [NSMutableDictionary new];

//    frameSubmit++;
//    NSLog(@"Analyzer Submitted frame %lu", frameSubmit);
    
    dispatch_group_t cpuAndGPUCompleted = dispatch_group_create();
    
    dispatch_group_enter(cpuAndGPUCompleted);

    dispatch_group_notify(cpuAndGPUCompleted, self.serialDictionaryQueue, ^{
        
//        frameComplete++;
//        NSLog(@"Analyer Completed frame %lu", frameComplete);

        if(completionHandler)
            completionHandler(dictionary, nil);
    });

#pragma mark - GPU Modules

    // Submit our GPU modules first, as they can upload and process while we then do work on the CPU.
    // Once we commit GPU work we can do CPU work, and then wait on both to complete

    if(self.gpuModules.count)
    {
        @autoreleasepool
        {
            dispatch_group_enter(cpuAndGPUCompleted);

            [frameCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
                dispatch_group_leave(cpuAndGPUCompleted);
            }];
            
//            for(GPUModule* module in self.gpuModules)

           [self.gpuModules enumerateObjectsUsingBlock:^(GPUModule * _Nonnull module, NSUInteger idx, BOOL * _Nonnull stop) {
               
               dispatch_group_enter(cpuAndGPUCompleted);

                SynopsisVideoFormat requiredFormat = [[module class] requiredVideoFormat];
                SynopsisVideoBacking requiredBacking = [[module class] requiredVideoBacking];
                SynopsisVideoFormatSpecifier* formatSpecifier = [[SynopsisVideoFormatSpecifier alloc] initWithFormat:requiredFormat backing:requiredBacking];
                
                id<SynopsisVideoFrame> currentFrame = [frameCache cachedFrameForFormatSpecifier:formatSpecifier];
                id<SynopsisVideoFrame> previousFrame = nil;
                
                if(self.lastFrameCache)
                    previousFrame = [self.lastFrameCache cachedFrameForFormatSpecifier:formatSpecifier];
                
                if(currentFrame)
                {
//                    NSLog(@"Analyzer got Frame: %@", currentFrame.label);

                    [module analyzedMetadataForCurrentFrame:currentFrame previousFrame:previousFrame commandBuffer:frameCommandBuffer completionBlock:^(NSDictionary *result, NSError *err) {
                        dispatch_barrier_sync(self.serialDictionaryQueue, ^{
                            
                            // If a module has a description key, we append, and not add to it
                            if(result[kSynopsisStandardMetadataDescriptionDictKey])
                            {
                                NSArray* cachedDescriptions = dictionary[kSynopsisStandardMetadataDescriptionDictKey];
                                
                                // this replaces our current description array with the new one
                                [dictionary addEntriesFromDictionary:result];
                                
                                // Re-write Description key with cached array appended to the new
                                dictionary[kSynopsisStandardMetadataDescriptionDictKey] = [dictionary[kSynopsisStandardMetadataDescriptionDictKey] arrayByAddingObjectsFromArray:cachedDescriptions];
                            }
                            else
                            {
                                [dictionary addEntriesFromDictionary:result];
                            }
                            
                            dispatch_group_leave(cpuAndGPUCompleted);

                        });
                    }];
                }
            }];
            
        }
    }
    
#pragma mark - CPU Modules
    
    if(self.cpuModules.count)
    {
        dispatch_group_enter(cpuAndGPUCompleted);

        NSBlockOperation* cpuCompletionOp = [NSBlockOperation blockOperationWithBlock:^{
            dispatch_group_leave(cpuAndGPUCompleted);
        }];
        
//        for(CPUModule* module in self.cpuModules)
        [self.cpuModules enumerateObjectsUsingBlock:^(CPUModule * _Nonnull module, NSUInteger idx, BOOL * _Nonnull stop) {
            SynopsisVideoFormat requiredFormat = [[module class] requiredVideoFormat];
            SynopsisVideoBacking requiredBacking = [[module class] requiredVideoBacking];
            SynopsisVideoFormatSpecifier* formatSpecifier = [[SynopsisVideoFormatSpecifier alloc] initWithFormat:requiredFormat backing:requiredBacking];
            
            id<SynopsisVideoFrame> currentFrame = [frameCache cachedFrameForFormatSpecifier:formatSpecifier];
            id<SynopsisVideoFrame> previousFrame = nil;
            
            if(self.lastFrameCache)
                previousFrame = [self.lastFrameCache cachedFrameForFormatSpecifier:formatSpecifier];
            
            if(currentFrame)
            {
                NSBlockOperation* moduleOperation = [NSBlockOperation blockOperationWithBlock:^{
                    
                    NSDictionary* result = [module analyzedMetadataForCurrentFrame:currentFrame previousFrame:previousFrame];
                    
                    dispatch_barrier_sync(self.serialDictionaryQueue, ^{
                        [dictionary addEntriesFromDictionary:result];
                    });
                }];
                
                NSString* key = NSStringFromClass([module class]);
                NSOperation* lastModuleOperation = self.lastModuleOperation[key];
                if(lastModuleOperation)
                {
                    [moduleOperation addDependency:lastModuleOperation];
                }
                
                self.lastModuleOperation[key] = moduleOperation;
                
                [cpuCompletionOp addDependency:moduleOperation];
                
                [self.moduleOperationQueues[idx] addOperation:moduleOperation];
            }
        }];

        [self.moduleOperationQueues[0] addOperation:cpuCompletionOp];
    }
    
//    if(self.gpuModules.count)
//    {
//        [frameCommandBuffer commit];
//        [frameCommandBuffer waitUntilCompleted];
//    }
    // Balance our first enter
    dispatch_group_leave(cpuAndGPUCompleted);

    self.lastFrameCache = frameCache;
}

#pragma mark - Finalization

- (NSDictionary*) finalizeMetadataAnalysisSessionWithError:(NSError**)error
{
    NSLog(@"FINALIZING ANALYZER !!?@?");
    NSMutableDictionary* finalized = [NSMutableDictionary new];
    
    for(CPUModule* module in self.cpuModules)
    {
        NSDictionary* moduleFinalMetadata = [module finaledAnalysisMetadata];
        
        // If a module has a description key, we append, and not add to it
        if(moduleFinalMetadata[kSynopsisStandardMetadataDescriptionDictKey])
        {
            NSArray* cachedDescriptions = finalized[kSynopsisStandardMetadataDescriptionDictKey];
            
            // this replaces our current description array with the new one
            [finalized addEntriesFromDictionary:moduleFinalMetadata];

            // Re-write Description key with cached array appended to the new
            finalized[kSynopsisStandardMetadataDescriptionDictKey] = [finalized[kSynopsisStandardMetadataDescriptionDictKey] arrayByAddingObjectsFromArray:cachedDescriptions];
        }
        else
        {
            [finalized addEntriesFromDictionary:moduleFinalMetadata];
        }
    }
    
    for(GPUModule* module in self.gpuModules)
    {
        NSDictionary* moduleFinalMetadata = [module finalizedAnalysisMetadata];
        
        // If a module has a description key, we append, and not add to it
        if(moduleFinalMetadata[kSynopsisStandardMetadataDescriptionDictKey])
        {
            NSArray* cachedDescriptions = finalized[kSynopsisStandardMetadataDescriptionDictKey];
            
            // this replaces our current description array with the new one
            [finalized addEntriesFromDictionary:moduleFinalMetadata];
            
            // Re-write Description key with cached array appended to the new
            finalized[kSynopsisStandardMetadataDescriptionDictKey] = [finalized[kSynopsisStandardMetadataDescriptionDictKey] arrayByAddingObjectsFromArray:cachedDescriptions];
        }
        else
        {
            [finalized addEntriesFromDictionary:moduleFinalMetadata];
        }
    }


    return finalized;
}



@end
