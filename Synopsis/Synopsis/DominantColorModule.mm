//
//  DominantColorAnalyzer.m
//  Synopsis
//
//  Created by vade on 11/10/16.
//  Copyright © 2016 metavisual. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import "SynopsisVideoFrameOpenCV.h"
#import "CIEDE2000.h"
#import "DominantColorModule.h"
#import "MedianCutOpenCV.hpp"
#import "SynopsisDenseFeature+Private.h"
//#import <Quartz/Quartz.h>

@interface DominantColorModule ()
{
    // for kMeans
    matType bestLables;
    matType centers;
}
@property (atomic, readwrite, strong) NSMutableArray<SynopsisDenseFeature*>* everyDominantColor;
@end

@implementation DominantColorModule

- (instancetype) initWithQualityHint:(SynopsisAnalysisQualityHint)qualityHint
{
    self = [super initWithQualityHint:qualityHint];
    {
        self.everyDominantColor = [NSMutableArray new];
    }
    return self;
}

- (NSString*) moduleName
{
    return kSynopsisStandardMetadataDominantColorValuesDictKey;//@"DominantColors";
}

+(SynopsisVideoBacking) requiredVideoBacking
{
    return SynopsisVideoBackingCPU;
}

+ (SynopsisVideoFormat) requiredVideoFormat
{
    return SynopsisVideoFormatPerceptual;
}

- (NSDictionary*) analyzedMetadataForCurrentFrame:(id<SynopsisVideoFrame>)frame previousFrame:(id<SynopsisVideoFrame>)lastFrame;
{
    SynopsisVideoFrameOpenCV* frameCV = (SynopsisVideoFrameOpenCV*)frame;

    // KMeans is slow as hell and also stochastic - same image run 2x gets slightly different results.
    // Median Cut is not particularly accurate ? Maybe I have a subtle bug due to averaging / scaling?
    // Dominant colors still average absed on centroid, even though we attempt to look up the closest
    // real color value near the centroid.
    
    // This needs some looking at and Median Cut is slow as fuck
    
    // result = [self dominantColorForCVMatKMeans:currentPerceptualImage];
    return [self dominantColorForCVMatMedianCutCV:frameCV.mat];
}

- (NSDictionary*) finaledAnalysisMetadata
{
    
    if(self.everyDominantColor.count == 0)
        return nil;
    
    // Also this code is heavilly borrowed so yea.
    int k = 5;
    int numPixels = (int)self.everyDominantColor.count;
    
    int sourceColorCount = 0;
    
    cv::Mat allDomColors = cv::Mat(1, numPixels, CV_32FC3);
    
    // Populate Median Cut Points by color values;
    for(NSArray* dominantColorsArray in self.everyDominantColor)
    {
        allDomColors.at<cv::Vec3f>(0, sourceColorCount) = cv::Vec3f([dominantColorsArray[0] floatValue], [dominantColorsArray[1] floatValue], [dominantColorsArray[2] floatValue]);
        sourceColorCount++;
    }
    
    bool useCIEDE2000 = USE_CIEDE2000;
    
    MedianCutOpenCV::ColorCube allColorCube(allDomColors, useCIEDE2000);
    
    auto palette = MedianCutOpenCV::medianCut(allColorCube, k, useCIEDE2000);
    
    NSMutableArray* dominantColors = [NSMutableArray new];
    NSMutableArray* dominantColorsLAB = [NSMutableArray new];

    for ( auto colorCountPair: palette )
    {
        // convert from LAB to BGR
        const cv::Vec3f& labColor = colorCountPair.first;
        
        cv::Mat closestLABPixel = cv::Mat(1,1, CV_32FC3, labColor);
        cv::Mat bgr(1,1, CV_32FC3);
        cv::cvtColor(closestLABPixel, bgr, FROM_PERCEPTUAL);
        
        [dominantColorsLAB addObject: [SynopsisDenseFeature valueWithCVMat:closestLABPixel]];
        
        cv::Vec3f bgrColor = bgr.at<cv::Vec3f>(0,0);
        
        [dominantColors addObject: @[@(bgrColor[2]),
                                     @(bgrColor[1]),
                                     @(bgrColor[0]),
                                     ]];
    }
    
    NSMutableDictionary* metadata = [NSMutableDictionary new];
    metadata[kSynopsisStandardMetadataDominantColorValuesDictKey] = dominantColors;
    metadata[kSynopsisStandardMetadataDescriptionDictKey] = [self matchColorNamesToLABColors:dominantColorsLAB];
    return metadata;
}

- (cv::Mat) nearestColorCIEDE2000:(cv::Vec3f)labColorVec3f inFrame:(matType)frame
{
    cv::Vec3f closestDeltaEColor;
    
    double delta = DBL_MAX;
    
    // iterate every pixel in our frame, and generate an CIEDE2000::LAB color from it
    // test the delta, and test if our pixel is our min
    cv::Mat frameMAT = frame;
    
    // Populate Median Cut Points by color values;
    for(int i = 0;  i < frameMAT.rows; i++)
    {
        for(int j = 0; j < frameMAT.cols; j++)
        {
            // get pixel value
            cv::Vec3f frameLABColor = frameMAT.at<cv::Vec3f>(i, j);
            
            double currentPixelDelta = CIEDE2000::CIEDE2000(labColorVec3f, frameLABColor);
            
            if(currentPixelDelta < delta)
            {
                closestDeltaEColor = frameLABColor;
                delta = currentPixelDelta;
            }
        }
    }
    
    cv::Mat closestLABColor(1,1, CV_32FC3, closestDeltaEColor);
    return closestLABColor;
}

// This doesnt appear to do anything.
- (cv::Mat) nearestColorMinMaxLoc:(cv::Vec3f)colorVec inFrame:(matType)frame
{
    //  find our nearest *actual* LAB pixel in the frame, not from the median cut..
    // Split image into channels
    std::vector<matType> frameChannels;
    cv::split(frame, frameChannels);
    
    // Find absolute differences for each channel
    matType diff_L;
    cv::absdiff(frameChannels[0], colorVec[0], diff_L);
    matType diff_A;
    cv::absdiff(frameChannels[1], colorVec[1], diff_A);
    matType diff_B;
    cv::absdiff(frameChannels[2], colorVec[2], diff_B);
    
    // Calculate L1 distance (diff_L + diff_A + diff_B)
    matType dist;
    matType dist2;
    cv::add(diff_L, diff_A, dist);
    cv::add(dist, diff_B, dist2);
    
    // Find the location of pixel with minimum color distance
    cv::Point minLoc;
    cv::minMaxLoc(dist2, 0, 0, &minLoc);
    
    // get pixel value
    cv::Vec3f closestColor = frame.at<cv::Vec3f>(minLoc);
    
    cv::Mat closestColorPixel(1,1, CV_32FC3, closestColor);
    
    return closestColorPixel;
}

- (NSDictionary*) dominantColorForCVMatMedianCutCV:(matType)image
{
    // Our Mutable Metadata Dictionary:
    NSMutableDictionary* metadata = [NSMutableDictionary new];
    
    // Also this code is heavilly borrowed so yea.
    int k = 5;
    
    bool useCIEDE2000 = USE_CIEDE2000;
    
    cv::Mat imageMat = image;
    
    auto palette = MedianCutOpenCV::medianCut(imageMat, k, useCIEDE2000);
    
    NSMutableArray* dominantColors = [NSMutableArray new];
    
    for ( auto colorCountPair: palette )
    {
        // convert from LAB to BGR
        const cv::Vec3f& labColor = colorCountPair.first;
        
        cv::Mat closestLABPixel = cv::Mat(1,1, CV_32FC3, labColor);
        
        // Looking at inspector output, its not clear that nearestColorMinMaxLoc is effective at all
        //        cv::Mat closestLABPixel = [self nearestColorMinMaxLoc:labColor inFrame:image];
        //        cv::Mat closestLABPixel = [self nearestColorCIEDE2000:labColor inFrame:image];
        
        // convert to BGR
        cv::Mat bgr(1,1, CV_32FC3);
        cv::cvtColor(closestLABPixel, bgr, FROM_PERCEPTUAL);
        
        cv::Vec3f bgrColor = bgr.at<cv::Vec3f>(0,0);
        
        NSArray* color = @[@(bgrColor[2]), // / 255.0), // R
                           @(bgrColor[1]), // / 255.0), // G
                           @(bgrColor[0]), // / 255.0), // B
                           ];

        [dominantColors addObject:color];

        SynopsisDenseFeature* labFeature = [SynopsisDenseFeature valueWithCVMat:closestLABPixel];
        
        // We will process this in finalize
        [self.everyDominantColor addObject:labFeature];
    }
    
    metadata[[self moduleName]] = dominantColors;
    
    return metadata;
    
}

- (NSDictionary*) dominantColorForCVMatKMeans:(matType)image
{
    // Our Mutable Metadata Dictionary:
    NSMutableDictionary* metadata = [NSMutableDictionary new];
    
    // We choose k = 5 to match Adobe Kuler because whatever.
    int k = 5;
    int n = image.rows * image.cols;
    
    std::vector<matType> imgSplit;
    cv::split(image,imgSplit);
    
    matType img3xN(n,3,CV_32F);
    
    for(int i = 0; i != 3; ++i)
    {
        imgSplit[i].reshape(1,n).copyTo(img3xN.col(i));
    }
    
    // TODO: figure out what the fuck makes sense here.
    cv::kmeans(img3xN,
               k,
               bestLables,
               //               cv::TermCriteria(),
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 5.0, 1.0),
               5,
               cv::KMEANS_PP_CENTERS,
               centers);
    
    NSMutableArray* dominantColors = [NSMutableArray new];
    
    //            cv::imshow("OpenCV Debug", quarterResLAB);
    
    for(int i = 0; i < centers.rows; i++)
    {
        // 0 1 or 0 - 255 .0 ?
        cv::Vec3f labColor = centers.at<cv::Vec3f>(i, 0);

        cv::Mat lab(1,1, CV_32FC3, cv::Vec3f(labColor[0], labColor[1], labColor[2]));
        
        cv::Mat bgr(1,1, CV_32FC3);
        
        cv::cvtColor(lab, bgr, FROM_PERCEPTUAL);
        
        cv::Vec3f bgrColor = bgr.at<cv::Vec3f>(0,0);
        
        NSArray* color = @[@(bgrColor[2]), // / 255.0), // R
                           @(bgrColor[1]), // / 255.0), // G
                           @(bgrColor[0]), // / 255.0), // B
                           ];
        
        SynopsisDenseFeature* labFeature = [SynopsisDenseFeature valueWithCVMat:lab];
        
        [dominantColors addObject:color];
        
        // We will process this in finalize
        [self.everyDominantColor addObject:labFeature];
    }
    
    metadata[@"DominantColors"] = dominantColors;
    metadata[@"Description"] = [self matchColorNamesToLABColors:dominantColors];

    return metadata;
}

#pragma mark - Color Helpers

-(NSArray*) matchColorNamesToLABColors:(NSArray<SynopsisDenseFeature*>*)labColorArray
{
    NSMutableSet* matchedNamedColors = [NSMutableSet setWithCapacity:labColorArray.count];
    
    for(SynopsisDenseFeature* color in labColorArray)
    {
        NSString* namedColor = [self closestNamedColorForLABColor:color];
//        NSLog(@"Found Color %@", namedColor);
        if(namedColor)
            [matchedNamedColors addObject:namedColor];
    }
    
    // Add our hack tag system:
    NSArray* colors = @[@"Colors:"];
    colors = [colors arrayByAddingObjectsFromArray: matchedNamedColors.allObjects];
    return colors;
}

- (NSString*) closestNamedColorForLABColor:(SynopsisDenseFeature*)color
{
    SynopsisDenseFeature* matchedColor = nil;

    SynopsisDenseFeature* white = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @1.0, @1.0, @1.0 ] ];
    SynopsisDenseFeature* black = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @0.0, @0.0, @0.0 ] ];
    SynopsisDenseFeature* gray = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @0.5, @0.5, @0.5 ] ];

    SynopsisDenseFeature* red = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @1.0, @0.0, @0.0 ] ];
    SynopsisDenseFeature* green = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @0.0, @1.0, @0.0 ] ];
    SynopsisDenseFeature* blue = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @0.0, @0.0, @1.0 ] ];
    
    SynopsisDenseFeature* cyan = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @0.0, @1.0, @1.0 ] ];
    SynopsisDenseFeature* magenta = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @1.0, @0.0, @1.0 ] ];
    SynopsisDenseFeature* yellow = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @1.0, @1.0, @0.0 ] ];

    SynopsisDenseFeature* orange = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @1.0, @0.5, @1.0 ] ];
    SynopsisDenseFeature* purple = [[SynopsisDenseFeature alloc] initWithFeatureArray:@[ @1.0, @0.0, @0.5 ] ];

    NSDictionary* knownColors = @{ @"White" : white, // White
                                   @"Black" : black, // Black
                                   @"Gray" : gray, // Gray
                                   @"Red" : red,
                                   @"Green" : green,
                                   @"Blue" : blue,
                                   @"Cyan" : cyan,
                                   @"Magenta" : magenta,
                                   @"Yellow" : yellow,
                                   @"Orange" : orange,
                                   @"Purple" : purple,
                                   };
    
    // Longest distance from any float color component
    float distance = FLT_MAX;
    
    for(SynopsisDenseFeature* namedColor in [knownColors allValues])
    {
        float newDistance = compareFeatureVector(namedColor, color);
        
        if(newDistance < distance)
        {
            distance = newDistance;
            matchedColor = namedColor;
        }
    }
    
    return [[knownColors allKeysForObject:matchedColor] firstObject];
}


@end
