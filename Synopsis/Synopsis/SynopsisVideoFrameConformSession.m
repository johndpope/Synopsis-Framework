//
//  SynopsisVideoFrameConformSession.m
//  Synopsis-Framework
//
//  Created by vade on 10/24/17.
//  Copyright © 2017 v002. All rights reserved.
//

#import "SynopsisVideoFrameConformSession.h"
#import "SynopsisVideoFrameConformHelperCPU.h"
#import "SynopsisVideoFrameConformHelperGPU.h"

@interface SynopsisVideoFrameConformSession ()
@property (readwrite, strong) SynopsisVideoFrameConformHelperCPU* conformCPUHelper;
@property (readwrite, strong) SynopsisVideoFrameConformHelperGPU* conformGPUHelper;

@property (readwrite, strong) NSSet<SynopsisVideoFormatSpecifier*>* cpuOnlyFormatSpecifiers;
@property (readwrite, strong) NSSet<SynopsisVideoFormatSpecifier*>* gpuOnlyFormatSpecifiers;

@property (readwrite, strong) id<MTLDevice>device;
@property (readwrite, strong) id<MTLCommandQueue> commandQueue;
@property (readwrite, strong) dispatch_semaphore_t inFlightBuffers;

@property (readwrite, strong) dispatch_queue_t serialCompletionQueue;

@end

@implementation SynopsisVideoFrameConformSession

- (instancetype) initWithRequiredFormatSpecifiers:(NSArray<SynopsisVideoFormatSpecifier*>*)formatSpecifiers device:(id<MTLDevice>)device inFlightBuffers:(NSUInteger)bufferCount
{
    self = [super init];
    if(self)
    {
        self.device = device;
        self.commandQueue = [self.device newCommandQueue];
        self.inFlightBuffers = dispatch_semaphore_create(bufferCount);

        self.conformCPUHelper = [[SynopsisVideoFrameConformHelperCPU alloc] initWithFlightBuffers:bufferCount];
        self.conformGPUHelper = [[SynopsisVideoFrameConformHelperGPU alloc] initWithCommandQueue:self.commandQueue inFlightBuffers:bufferCount];

        self.serialCompletionQueue = dispatch_queue_create("info.synopsis.formatConversion", DISPATCH_QUEUE_SERIAL);
        
        NSMutableSet<SynopsisVideoFormatSpecifier*>* cpu = [NSMutableSet new];
        NSMutableSet<SynopsisVideoFormatSpecifier*>* gpu = [NSMutableSet new];
        
        for(SynopsisVideoFormatSpecifier* format in formatSpecifiers)
        {
            switch(format.backing)
            {
                case SynopsisVideoBackingGPU:
                    [gpu addObject:format];
                    break;
                case SynopsisVideoBackingCPU:
                    [cpu addObject:format];
                    break;
                case SynopsisVideoBackingNone:
                    break;
            }
        }
        
        self.cpuOnlyFormatSpecifiers = cpu;
        self.gpuOnlyFormatSpecifiers = gpu;
    }
    
    return self;
}

- (void) conformPixelBuffer:(CVPixelBufferRef)pixelBuffer atTime:(CMTime)time withTransform:(CGAffineTransform)transform rect:(CGRect)rect               
 completionBlock:(SynopsisVideoFrameConformSessionCompletionBlock)completionBlock
{    
    // Because we have 2 different completion blocks we must coalesce into one, we use
    // dispatch notify to tell us when we are actually done.
    
    id<MTLCommandBuffer> commandBuffer = self.commandQueue.commandBuffer;

    NSArray<SynopsisVideoFormatSpecifier*>* localCPUFormats = [self.cpuOnlyFormatSpecifiers allObjects];
    NSArray<SynopsisVideoFormatSpecifier*>* localGPUFormats = [self.gpuOnlyFormatSpecifiers allObjects];

    SynopsisVideoFrameCache* allFormatCache = [[SynopsisVideoFrameCache alloc] init];
    
    dispatch_group_t formatConversionGroup = dispatch_group_create();
    dispatch_group_enter(formatConversionGroup);
    
//    __block SynopsisVideoFrameCache* cpuCache = nil;
//    __block NSError* cpuError = nil;
//
//    __block SynopsisVideoFrameCache* gpuCache = nil;
//    __block NSError* gpuError = nil;
    
    dispatch_group_notify(formatConversionGroup, self.serialCompletionQueue, ^{
        
        if(completionBlock)
        {
            completionBlock(commandBuffer, allFormatCache, nil);
            
            dispatch_semaphore_signal(self.inFlightBuffers);
        }
    });
    
    dispatch_semaphore_wait(self.inFlightBuffers, DISPATCH_TIME_FOREVER);
    
    if(localGPUFormats.count)
    {
        dispatch_group_enter(formatConversionGroup);
        [self.conformGPUHelper conformPixelBuffer:pixelBuffer
                                           atTime:time
                                        toFormats:localGPUFormats
                                    withTransform:transform
                                             rect:rect
                                    commandBuffer:commandBuffer
                                  completionBlock:^(id<MTLCommandBuffer> commandBuffer, SynopsisVideoFrameCache * gpuCache, NSError *err) {
                                      
                                      for(SynopsisVideoFormatSpecifier* format in localGPUFormats)
                                      {
                                          id<SynopsisVideoFrame> frame = [gpuCache cachedFrameForFormatSpecifier:format];
                                          
                                          if(frame)
                                          {
                                              [allFormatCache cacheFrame:frame];
                                          }
                                      }
                                      
                                      dispatch_group_leave(formatConversionGroup);
                                  }];
    }
    
    if(localCPUFormats.count)
    {
        dispatch_group_enter(formatConversionGroup);
        [self.conformCPUHelper conformPixelBuffer:pixelBuffer
                                           atTime:time
                                        toFormats:localCPUFormats
                                    withTransform:transform
                                             rect:rect
                                  completionBlock:^(id<MTLCommandBuffer> commandBuffer, SynopsisVideoFrameCache * cpuCache, NSError *err) {

                                      for(SynopsisVideoFormatSpecifier* format in localCPUFormats)
                                      {
                                          id<SynopsisVideoFrame> frame = [cpuCache cachedFrameForFormatSpecifier:format];
                                          
                                          if(frame)
                                          {
                                              [allFormatCache cacheFrame:frame];
                                          }
                                      }
                                      
                                      dispatch_group_leave(formatConversionGroup);
                                  }];
    }

    dispatch_group_leave(formatConversionGroup);
}


- (void) blockForPendingConforms
{
    [self.conformCPUHelper.conformQueue waitUntilAllOperationsAreFinished];
}

- (void) cancelPendingConforms
{
    [self.conformCPUHelper.conformQueue cancelAllOperations];
}


@end
