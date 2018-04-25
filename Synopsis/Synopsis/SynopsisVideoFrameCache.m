//
//  SynopsisVideoFrameCache.m
//  Synopsis-Framework
//
//  Created by vade on 10/24/17.
//  Copyright © 2017 v002. All rights reserved.
//

#import "SynopsisVideoFrameCache.h"

@interface SynopsisVideoFrameCache ()
@property (readwrite, strong) NSMutableArray* videoCacheArray;
@property (readwrite, strong) NSLock* arrayLock;
@end
@implementation SynopsisVideoFrameCache

- (instancetype) init
{
    self = [super init];
    if(self)
    {
//        @synchronized(self)
        {
            self.videoCacheArray = [NSMutableArray new];
            self.arrayLock = [[NSLock alloc] init];
        }
    }
    return self;
}

- (void) cacheFrame:(id<SynopsisVideoFrame>)frame
{
//    @synchronized(self)
    {
        [self.arrayLock lock];
        [self.videoCacheArray addObject:frame];
        [self.arrayLock unlock];
    }
}

- (id<SynopsisVideoFrame>) cachedFrameForFormatSpecifier:(SynopsisVideoFormatSpecifier*)formatSpecifier;
{
//    @synchronized(self)
    {
        [self.arrayLock lock];
        id<SynopsisVideoFrame> matchingFrame = nil;
        for(id<SynopsisVideoFrame>frame in self.videoCacheArray)
        {
            if( [frame.videoFormatSpecifier isEqual:formatSpecifier])
            {
                matchingFrame = frame;
                break;
            }
        }
        
        [self.arrayLock unlock];
        return matchingFrame;
    }
}


@end
