//
//  SynopsisVideoFrameConformHelperGPU.m
//  Synopsis-Framework
//
//  Created by vade on 10/24/17.
//  Copyright © 2017 v002. All rights reserved.
//

#import "SynopsisVideoFrameConformHelperGPU.h"
#import "SynopsisVideoFrameMPImage.h"

#import <CoreImage/CoreImage.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Metal/Metal.h>

@interface SynopsisVideoFrameConformHelperGPU ()
{
    CVMetalTextureCacheRef textureCacheRef;
}
//@property (readwrite, strong) NSOperationQueue* conformQueue;

@property (readwrite, strong) id<MTLCommandQueue>commandQueue;
@property (readwrite, strong) MPSImageConversion* imageConversion;
@property (readwrite, strong) MPSImageBilinearScale* scaleForCoreML;


@end

@implementation SynopsisVideoFrameConformHelperGPU
- (instancetype) initWithCommandQueue:(id<MTLCommandQueue>)queue inFlightBuffers:(NSUInteger)bufferCount;

{
    self = [super init];
    if(self)
    {
        self.commandQueue = queue;
        
        CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, self.commandQueue.device, NULL, &textureCacheRef);
        self.scaleForCoreML = [[MPSImageBilinearScale alloc] initWithDevice:self.commandQueue.device];
    }
    
    return self;
}

- (void) dealloc
{
    if(textureCacheRef)
    {
        CVMetalTextureCacheFlush(textureCacheRef, 0);
        CFRelease(textureCacheRef);
        textureCacheRef = NULL;
    }
}


static NSUInteger frameSubmit = 0;
static NSUInteger frameComplete = 0;

- (void) conformPixelBuffer:(CVPixelBufferRef)pixelBuffer
                  toFormats:(NSArray<SynopsisVideoFormatSpecifier*>*)formatSpecifiers
              withTransform:(CGAffineTransform)transform
                       rect:(CGRect)destinationRect
              commandBuffer:(id<MTLCommandBuffer>)commandBuffer
            completionBlock:(SynopsisVideoFrameConformSessionCompletionBlock)completionBlock;
{
    frameSubmit++;

    id<MTLCommandBuffer> conformBuffer = self.commandQueue.commandBuffer;
    
//    NSBlockOperation* conformOperation = [NSBlockOperation blockOperationWithBlock:^{

        CVPixelBufferRetain(pixelBuffer);
        
        // Create our metal texture from our CVPixelBuffer
        size_t width = CVPixelBufferGetWidth(pixelBuffer);
        size_t height = CVPixelBufferGetHeight(pixelBuffer);
        
        CVMetalTextureRef inputCVTexture = NULL;
        CVMetalTextureCacheFlush(textureCacheRef, 0);
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCacheRef, pixelBuffer, NULL, MTLPixelFormatBGRA8Unorm, width, height, 0, &inputCVTexture);
        
        assert(inputCVTexture != NULL);
        
        id<MTLTexture> inputMTLTexture = CVMetalTextureGetTexture(inputCVTexture);
        
        assert(inputMTLTexture != NULL);
        
        MPSImage* sourceInput = [[MPSImage alloc] initWithTexture:inputMTLTexture featureChannels:3];
        sourceInput.label = [NSString stringWithFormat:@"%@, %lu", @"Source", (unsigned long)frameSubmit];
        
#pragma mark - Convert :
        
//        if(self.imageConversion == nil)
//        {
//            CGColorSpaceRef source = CVImageBufferGetColorSpace(pixelBuffer);
//            CGColorSpaceRef destination = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGBLinear);
//            //        CGColorSpaceRef destination = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
//            source = CGColorSpaceRetain(source);
//            BOOL deleteSource = NO;
//            if(source == NULL)
//            {
//                // Assume video is HD color space if not otherwise marked
//                source = CGColorSpaceCreateWithName(kCGColorSpaceITUR_709);
//                deleteSource = YES;
//            }
//
//            CGColorConversionInfoRef colorConversionInfo = CGColorConversionInfoCreate(source, destination);
//
//            CGFloat background[4] = {0,0,0,0};
//            self.imageConversion = [[MPSImageConversion alloc] initWithDevice:self.device
//                                                                     srcAlpha:MPSAlphaTypeAlphaIsOne
//                                                                    destAlpha:MPSAlphaTypeAlphaIsOne
//                                                              backgroundColor:background
//                                                               conversionInfo:colorConversionInfo];
//
//            if(deleteSource)
//                CGColorSpaceRelease(source);
//        }
//
//        MPSImageDescriptor* convertDescriptor = [[MPSImageDescriptor alloc] init];
//        convertDescriptor.width = sourceInput.width;
//        convertDescriptor.height = sourceInput.height;
//        convertDescriptor.featureChannels = sourceInput.featureChannels;
//        convertDescriptor.numberOfImages = 1;
//        convertDescriptor.channelFormat = MPSImageFeatureChannelFormatUnorm8;
//        convertDescriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
//
//        //    MPSImageDescriptor* convertDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatUnorm8
//        //                                                                                           width:sourceInput.width
//        //                                                                                          height:sourceInput.height
//        //                                                                                 featureChannels:sourceInput.featureChannels];
//        //    convertDescriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
//
//        MPSImage* convertTarget = [[MPSImage alloc] initWithDevice:self.device imageDescriptor:convertDescriptor];
//        convertTarget.label = [NSString stringWithFormat:@"%@, %lu", @"Convert", (unsigned long)frameSubmit];
//
//        [self.imageConversion encodeToCommandBuffer:commandBuffer sourceImage:sourceInput destinationImage:convertTarget];
    
#pragma mark - Resize :
        
            MPSImageDescriptor* resizeDescriptor = [[MPSImageDescriptor alloc] init];
            resizeDescriptor.width = destinationRect.size.width;
            resizeDescriptor.height = destinationRect.size.height;
            resizeDescriptor.featureChannels = 3;
            resizeDescriptor.numberOfImages = 1;
            resizeDescriptor.channelFormat = MPSImageFeatureChannelFormatUnorm8;
            resizeDescriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    
    MPSImage* resizeTarget = [[MPSImage alloc] initWithDevice:self.commandQueue.device imageDescriptor:resizeDescriptor];
            resizeTarget.label = [NSString stringWithFormat:@"%@, %lu", @"Resize", (unsigned long)frameSubmit];
    
            [self.scaleForCoreML encodeToCommandBuffer:conformBuffer sourceImage:sourceInput destinationImage:resizeTarget];
    
    [conformBuffer addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
    
            if(completionBlock)
            {
                frameComplete++;
//                NSLog(@"Conform Completed frame %lu", frameComplete);
                SynopsisVideoFrameCache* cache = [[SynopsisVideoFrameCache alloc] init];
                SynopsisVideoFormatSpecifier* resultFormat = [[SynopsisVideoFormatSpecifier alloc] initWithFormat:SynopsisVideoFormatBGR8 backing:SynopsisVideoBackingGPU];
                SynopsisVideoFrameMPImage* result = [[SynopsisVideoFrameMPImage alloc] initWithMPSImage:resizeTarget formatSpecifier:resultFormat];
                
                [cache cacheFrame:result];
                
                completionBlock(commandBuffer, cache, nil);
                
                //            if(deleteSource)
                //                CGColorSpaceRelease(source);
                
                // Release our CVMetalTextureRef
                CFRelease(inputCVTexture);
                
                // We always have to release our pixel buffer
                CVPixelBufferRelease(pixelBuffer);
            }
        }];
    
        [conformBuffer commit];
    
//    }];
//    
//    [self.conformQueue addOperations:@[conformOperation] waitUntilFinished:NO];

}


@end
