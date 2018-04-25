//
//  MPSMobileNetFeatureExtractor.m
//  Synopsis-macOS
//
//  Created by vade on 10/27/17.
//  Copyright © 2017 v002. All rights reserved.
//

#import <Vision/Vision.h>

// Apple Model:
//#import "MobileNet.h"

// Our Models + Classifiers
#import "CinemaNetFeatureExtractor.h"
#import "CinemaNetShotAnglesClassifier.h"
#import "CinemaNetShotFramingClassifier.h"
#import "CinemaNetShotSubjectClassifier.h"
#import "CinemaNetShotTypeClassifier.h"
#import "PlacesNetClassifier.h"

#import "GPUVisionMobileNet.h"

#import "SynopsisSlidingWindow.h"

@interface GPUVisionMobileNet ()
{
    CGColorSpaceRef linear;
}

@property (readwrite, strong) CIContext* context;
@property (readwrite, strong) VNSequenceRequestHandler* sequenceRequestHandler;

@property (readwrite, strong) VNCoreMLModel* cinemaNetCoreVNModel;
@property (readwrite, strong) CinemaNetFeatureExtractor* cinemaNetCoreMLModel;
@property (readwrite, strong) CinemaNetShotAnglesClassifier* cinemaNetShotAnglesClassifierMLModel;
@property (readwrite, strong) CinemaNetShotFramingClassifier* cinemaNetShotFramingClassifierMLModel;
@property (readwrite, strong) CinemaNetShotSubjectClassifier* cinemaNetShotSubjectClassifierMLModel;
@property (readwrite, strong) CinemaNetShotTypeClassifier* cinemaNetShotTypeClassifierMLModel;
@property (readwrite, strong) PlacesNetClassifier* placesNetClassifierMLModel;

@property (readwrite, strong) NSMutableArray<NSNumber*>* averageFeatureVec;
@property (readwrite, strong) NSMutableArray<SynopsisDenseFeature*>* windowAverages;
@property (readwrite, strong) NSMutableArray<NSValue*>* windowAverageTimes;
@property (readwrite, strong) NSArray* labels;

//_Nullable@property (readwrite, strong) NSMutableArray<SynopsisDenseFeature*> slidingWindowAverage;

//@property (readwrite, strong) SynopsisSlidingWindow* windowA;
//@property (readwrite, strong) SynopsisSlidingWindow* windowB;

@property (readwrite, strong) NSMutableArray<SynopsisSlidingWindow*>* windows;


@end

const NSUInteger stride = 5;
const NSUInteger numWindows = 2;

@implementation GPUVisionMobileNet

// GPU backed modules init with an options dict for Metal Device bullshit
- (instancetype) initWithQualityHint:(SynopsisAnalysisQualityHint)qualityHint device:(id<MTLDevice>)device
{
    self = [super initWithQualityHint:qualityHint device:device];
    if(self)
    {
        self.windowAverages = [NSMutableArray new];
        self.windowAverageTimes = [NSMutableArray new];
        self.windows =[NSMutableArray new];

        for(NSUInteger i = 0; i < numWindows; i++)
        {
            SynopsisSlidingWindow* aWindow = [[SynopsisSlidingWindow alloc] initWithLength:10 offset:stride * i];
            [self.windows addObject:aWindow];
        }
    
        
        linear = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGBLinear);

        NSDictionary* opt = @{ kCIContextWorkingColorSpace : (__bridge id)linear,
                               kCIContextOutputColorSpace : (__bridge id)linear,
                                };
        self.context = [CIContext contextWithMTLDevice:device options:opt];
        self.sequenceRequestHandler = [[VNSequenceRequestHandler alloc] init];

        NSError* error = nil;
        self.cinemaNetCoreMLModel = [[CinemaNetFeatureExtractor alloc] init];
        self.cinemaNetShotAnglesClassifierMLModel = [[CinemaNetShotAnglesClassifier alloc] init];
        self.cinemaNetShotFramingClassifierMLModel = [[CinemaNetShotFramingClassifier alloc] init];
        self.cinemaNetShotSubjectClassifierMLModel = [[CinemaNetShotSubjectClassifier alloc] init];
        self.cinemaNetShotTypeClassifierMLModel = [[CinemaNetShotTypeClassifier alloc] init];
        self.placesNetClassifierMLModel = [[PlacesNetClassifier alloc] init];
        
        self.cinemaNetCoreVNModel = [VNCoreMLModel modelForMLModel:self.cinemaNetCoreMLModel.model error:&error];
        
        if(error)
        {
            NSLog(@"Error: %@", error);
        }
        
    }
    return self;
}

- (void)dealloc
{
    CGColorSpaceRelease(linear);
}

- (NSString*) moduleName
{
    return kSynopsisStandardMetadataFeatureVectorDictKey;
}

+ (SynopsisVideoBacking) requiredVideoBacking
{
    return SynopsisVideoBackingGPU;
}

+ (SynopsisVideoFormat) requiredVideoFormat
{
    return SynopsisVideoFormatBGR8;
}

- (void) beginAndClearCachedResults
{
    self.averageFeatureVec = nil;
}

- (void) analyzedMetadataForCurrentFrame:(id<SynopsisVideoFrame>)frame previousFrame:(id<SynopsisVideoFrame>)lastFrame commandBuffer:(id<MTLCommandBuffer>)buffer completionBlock:(GPUModuleCompletionBlock)completionBlock;
{
    SynopsisVideoFrameMPImage* frameMPImage = (SynopsisVideoFrameMPImage*)frame;
    
    CIImage* imageForRequest = [CIImage imageWithMTLTexture:frameMPImage.mpsImage.texture options:nil];
    
    VNCoreMLRequest* mobileRequest = [[VNCoreMLRequest alloc] initWithModel:self.cinemaNetCoreVNModel completionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
                
        NSMutableDictionary* metadata = nil;
        if([request results].count)
        {
            VNCoreMLFeatureValueObservation* featureOutput = [[request results] firstObject];
            MLMultiArray* featureVector = featureOutput.featureValue.multiArrayValue;
            
            NSMutableArray<NSNumber*>*vec = [NSMutableArray new];
            
            if(self.averageFeatureVec == nil)
            {
                for(NSUInteger i = 0; i < featureVector.count; i++)
                {
                    vec[i] = featureVector[i];
                }
                
                self.averageFeatureVec = vec;
            }
            
            else
            {
                for(NSUInteger i = 0; i < featureVector.count; i++)
                {
                    NSNumber* avgFeatureValue = self.averageFeatureVec[i];
                    NSNumber* featureValue = featureVector[i];
                    
                    self.averageFeatureVec[i] = @( (avgFeatureValue.floatValue + featureValue.floatValue) * 0.5 );
                    vec[i] = featureValue;
                }
            }
            
            SynopsisDenseFeature* denseFeatureVector = [[SynopsisDenseFeature alloc] initWithFeatureArray:vec];

            metadata = [NSMutableDictionary dictionary];

            [self.windows enumerateObjectsUsingBlock:^(SynopsisSlidingWindow * _Nonnull window, NSUInteger idx, BOOL * _Nonnull stop) {
                SynopsisDenseFeature* possible = [window appendFeature:denseFeatureVector];
                if(possible != nil)
                {
                    [self.windowAverages addObject:possible];
                    [self.windowAverageTimes addObject:[NSValue valueWithCMTime:frame.presentationTimeStamp]];
                }
            }];
            
            __block CinemaNetShotAnglesClassifierOutput* anglesOutput = nil;
            __block CinemaNetShotFramingClassifierOutput* framingOutput = nil;
            __block CinemaNetShotSubjectClassifierOutput* subjectOutput = nil;
            __block CinemaNetShotTypeClassifierOutput* typeOutput = nil;
            __block PlacesNetClassifierOutput* placesOutput = nil;
            
            dispatch_group_t classifierGroup = dispatch_group_create();
            
            dispatch_group_enter(classifierGroup);
            
            dispatch_group_notify(classifierGroup, self.completionQueue, ^{
                
                NSString* topAngleLabel = anglesOutput.classLabel;
                NSString* topFrameLabel = framingOutput.classLabel;
                NSString* topSubjectLabel = subjectOutput.classLabel;
                NSString* topTypeLabel = typeOutput.classLabel;
                NSString* placesNetLabel = placesOutput.classLabel;
                
                topAngleLabel = [topAngleLabel capitalizedString];
                topFrameLabel = [topFrameLabel capitalizedString];
                topSubjectLabel = [topSubjectLabel capitalizedString];
                topTypeLabel = [topTypeLabel capitalizedString];
                placesNetLabel = [placesNetLabel capitalizedString];

                NSMutableArray<NSString*>* labels = [NSMutableArray new];
                
                if(topAngleLabel)
                {
                    [labels addObject:@"Shot Angle:"];
                    [labels addObject:topAngleLabel];
                }
                if(topFrameLabel)
                {
                    [labels addObject:@"Shot Framing:"];
                    [labels addObject:topFrameLabel];
                }
                if(topSubjectLabel)
                {
                    [labels addObject:@"Shot Subject:"];
                    [labels addObject:topSubjectLabel];
                }
                if(topTypeLabel)
                {
                    [labels addObject:@"Shot Type:"];
                    [labels addObject:topTypeLabel];
                }
//                if(imageNetLabel)
//                {
//                    [labels addObject:@"Objects:"];
//                    [labels addObjectsFromArray:imageNetLabel];
//                }
                if(placesNetLabel)
                {
                    [labels addObject:@"Location:"];
                    [labels addObject:placesNetLabel];
                }

                metadata[kSynopsisStandardMetadataFeatureVectorDictKey] = vec;
                metadata[kSynopsisStandardMetadataDescriptionDictKey] = labels;

                self.labels = labels;

                if(completionBlock)
                {
                    completionBlock(metadata, nil);
                }
            });
            
            // If we have a valid feature vector result, parallel classify.
            
            dispatch_group_enter(classifierGroup);
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
                anglesOutput = [self.cinemaNetShotAnglesClassifierMLModel predictionFromInput_1__BottleneckInputPlaceholder__0:featureVector  error:nil];
                dispatch_group_leave(classifierGroup);
            });
            
            dispatch_group_enter(classifierGroup);
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
                framingOutput = [self.cinemaNetShotFramingClassifierMLModel predictionFromInput_1__BottleneckInputPlaceholder__0:featureVector  error:nil];
                dispatch_group_leave(classifierGroup);
            });
            
            dispatch_group_enter(classifierGroup);
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
                subjectOutput = [self.cinemaNetShotSubjectClassifierMLModel predictionFromInput_1__BottleneckInputPlaceholder__0:featureVector  error:nil];
                dispatch_group_leave(classifierGroup);
            });
            
            dispatch_group_enter(classifierGroup);
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
                typeOutput = [self.cinemaNetShotTypeClassifierMLModel predictionFromInput_1__BottleneckInputPlaceholder__0:featureVector  error:nil];
                dispatch_group_leave(classifierGroup);
            });
            
            dispatch_group_enter(classifierGroup);
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
                placesOutput = [self.placesNetClassifierMLModel predictionFromInput_1__BottleneckInputPlaceholder__0:featureVector  error:nil];
                dispatch_group_leave(classifierGroup);
            });
            
            dispatch_group_leave(classifierGroup);
        }
    }];
    
    mobileRequest.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
    mobileRequest.preferBackgroundProcessing = NO;

    // Works fine:
    CGImagePropertyOrientation orientation = kCGImagePropertyOrientationDownMirrored;
    VNImageRequestHandler* imageRequestHandler = [[VNImageRequestHandler alloc] initWithCIImage:imageForRequest orientation:orientation options:@{}];

    NSError* submitError = nil;
    if(![imageRequestHandler performRequests:@[mobileRequest] error:&submitError] )
//    if(![self.sequenceRequestHandler performRequests:@[mobileNetRequest] onCIImage:imageForRequest error:&submitError])
    {
        NSLog(@"Error submitting request: %@", submitError);
    }
}

- (NSDictionary*) finalizedAnalysisMetadata;
{
    NSMutableArray* windowAverages = [NSMutableArray arrayWithCapacity:self.windowAverages.count];
    
    [self.windowAverages enumerateObjectsUsingBlock:^(SynopsisDenseFeature * _Nonnull feature, NSUInteger idx, BOOL * _Nonnull stop) {
        NSValue* windowTime = [self.windowAverageTimes objectAtIndex:idx];

        [windowAverages addObject: @{ @"Feature" : [feature arrayValue],
                                      @"Time" : (NSDictionary*) CFBridgingRelease(CMTimeCopyAsDictionary([windowTime CMTimeValue], kCFAllocatorDefault)),
                                      }];
    }];
    
    return @{
             kSynopsisStandardMetadataFeatureVectorDictKey : (self.averageFeatureVec) ? self.averageFeatureVec : @[ ],
             kSynopsisStandardMetadataInterestingFeaturesAndTimesDictKey  : (windowAverages) ? windowAverages : @[ ],
             kSynopsisStandardMetadataDescriptionDictKey: (self.labels) ? self.labels : @[ ],
             };
}




@end
