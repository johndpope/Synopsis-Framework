//
//  SynopsisSlidingWindow.m
//  Synopsis-macOS
//
//  Created by vade on 4/23/18.
//  Copyright © 2018 v002. All rights reserved.
//
#import <opencv2/opencv.hpp>

#import "SynopsisSlidingWindow.h"
#import "SynopsisDenseFeature+Private.h"

@interface SynopsisSlidingWindow ()
@property (readwrite, assign) NSUInteger length;
@property (readwrite, assign) NSUInteger offset;

@property (readwrite, assign) NSUInteger appendAttempts;

@property (readwrite, strong) NSMutableArray<SynopsisDenseFeature*>*features;


@end

@implementation SynopsisSlidingWindow
- (instancetype _Nonnull) initWithLength:(NSUInteger)len offset:(NSUInteger)offset
{
    self = [super init];
    if(self)
    {
        self.length = len;
        self.offset = offset;
        self.appendAttempts = offset;
        self.features = [NSMutableArray new];
    }
    return self;
}


- (nullable SynopsisDenseFeature*)appendFeature:(SynopsisDenseFeature*)feature
{
    if(self.appendAttempts)
    {
        self.appendAttempts--;
    }
    else
    {
        [self.features addObject:feature];
    }
        
    if(self.features.count == self.length)
    {
        // compute averages of our len features
        
        cv::Mat averageFeature = cv::Mat(self.features.firstObject.featureCount, 1 , CV_32FC1, (float)0.0f);
        for(SynopsisDenseFeature* featureVector in self.features)
        {
            cv::Mat featureMat = [featureVector cvMatValue];
            
            cv::add(averageFeature, featureMat, averageFeature);
        }

        cv::divide(averageFeature, (double)self.features.count, averageFeature);
        
        [self.features removeAllObjects];
        
        return [SynopsisDenseFeature valueWithCVMat:averageFeature];
    }

    return nil;
}

- (NSUInteger) count
{
    return self.features.count;
}


@end

