//
//  SynopsisVideoFrameConformHelperGPU.h
//  Synopsis-Framework
//
//  Created by vade on 10/24/17.
//  Copyright © 2017 v002. All rights reserved.
//

#import <CoreVideo/CoreVideo.h>
#import "SynopsisVideoFrameCache.h"
#import "SynopsisVideoFrameConformSession.h"

@interface SynopsisVideoFrameConformHelperGPU : NSObject

- (instancetype) initWithCommandQueue:(id<MTLCommandQueue>)queue inFlightBuffers:(NSUInteger)bufferCount;

- (void) conformPixelBuffer:(CVPixelBufferRef)pixelbuffer
                     atTime:(CMTime)time
                  toFormats:(NSArray<SynopsisVideoFormatSpecifier*>*)formatSpecifiers
              withTransform:(CGAffineTransform)transform
                       rect:(CGRect)rect
              commandBuffer:(id<MTLCommandBuffer>)commandBuffer
            completionBlock:(SynopsisVideoFrameConformSessionCompletionBlock)completionBlock;

@end
