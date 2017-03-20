//
//  NSSortDescriptor+Synopsis_NSSortDescriptor.h
//  Synopsis-Framework
//
//  Created by vade on 8/5/16.
//  Copyright © 2016 v002. All rights reserved.
//

#import <Cocoa/Cocoa.h>

@interface NSSortDescriptor (SynopsisMetadata)

// Uses weights best match of all the independed sorting / weighting algorithms
+ (NSSortDescriptor*)synopsisBestMatchSortDescriptorRelativeTo:(NSDictionary*)standardMetadata;

// See which two objects are closest to the relativeHash
+ (NSSortDescriptor*)synopsisFeatureSortDescriptorRelativeTo:(NSArray*)featureVector;

// See which two objects are closest to the relativeHash
+ (NSSortDescriptor*)synopsisHashSortDescriptorRelativeTo:(NSString*)relativeHash;

// See which two objects are closest to the relativeHash
+ (NSSortDescriptor*)synopsisMotionVectorSortDescriptorRelativeTo:(NSArray*)motionVector;

// Use OpenCV Histogram Comparison
+ (NSSortDescriptor*)synopsisHistogramSortDescriptorRelativeTo:(NSArray*)histogram;

// Dominant Color RGB similarity
+ (NSSortDescriptor*)synopsisDominantRGBDescriptorRelativeTo:(NSArray*)colors;

// Dominant Color HSB similarity
+ (NSSortDescriptor*)synopsisDominantHSBDescriptorRelativeTo:(NSArray*)colors;


// Todo: Use CIE Delta E 2000 / 1994 and 1976
+ (NSSortDescriptor*)synopsisColorCIESortDescriptorRelativeTo:(NSColor*)color;

// Sort Color by Hue
+ (NSSortDescriptor*)synopsisColorHueSortDescriptor;

// Sort Color by Saturation
+ (NSSortDescriptor*)synopsisColorSaturationSortDescriptor;

// Sort Color by Brightness
+ (NSSortDescriptor*)synopsisColorBrightnessSortDescriptor;



@end
