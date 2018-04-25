//
//  SynopsisSlidingWindow.h
//  Synopsis-macOS
//
//  Created by vade on 4/23/18.
//  Copyright © 2018 v002. All rights reserved.
//

#import <Foundation/Foundation.h>

@class SynopsisDenseFeature;

@interface SynopsisSlidingWindow : NSObject
@property (readonly, assign) NSUInteger length;
@property (readonly, assign) NSUInteger offset;

- (instancetype _Nonnull) initWithLength:(NSUInteger)len offset:(NSUInteger)offset;

- (SynopsisDenseFeature* _Nullable)appendFeature:(SynopsisDenseFeature* _Nonnull )feature;

- (NSUInteger) count;

@end
