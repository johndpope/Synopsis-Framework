//
//  SynopsisSlidingWindow.m
//  Synopsis-macOS
//
//  Created by vade on 4/23/18.
//  Copyright © 2018 v002. All rights reserved.
//

#import "SynopsisSlidingWindow.h"
#import "SynopsisDenseFeature.h"

@interface SynopsisSlidingWindow ()
@property (readwrite, assign) NSUInteger len;
@property (readwrite, strong) NSMutableArray<SynopsisDenseFeature*>*features;


@end

@implementation SynopsisSlidingWindow
- (instancetype) initWithLength:(NSUInteger)len;
{
    self = [super init];
    if(self)
    {
        self.len = len;
        self.features = [NSMutableArray new];
    }
    return self;
}


- (nullable NSArray<SynopsisDenseFeature*>*)appendFeature:(SynopsisDenseFeature*)feature
{
    if(self.features.count > self.len)
    {
        // compute averages of our len features
        
        for(SynopsisDenseFeature* feature in self.features)
        {
            
        }
        
        
        [self.features removeAllObjects];
        
        return features;
    }
    else
    {
        [self.features addObject:feature];
    }
}


@end
