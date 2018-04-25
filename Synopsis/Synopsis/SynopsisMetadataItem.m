//
//  SynopsisMetadataItem.m
//  Synopslight
//
//  Created by vade on 7/28/16.
//  Copyright © 2016 v002. All rights reserved.
//

#import <Synopsis/Synopsis.h>
#import <AVFoundation/AVFoundation.h>
#import "SynopsisMetadataItem.h"

#import "Color+linearRGBColor.h"

@interface SynopsisMetadataItem ()
{
    CGImageRef cachedImage;
}
@property (readwrite) NSURL* url;
@property (readwrite, strong) AVAsset* asset;
@property (readwrite, strong) NSDictionary* globalSynopsisMetadata;
@property (readwrite, strong) SynopsisMetadataDecoder* decoder;
@end

@implementation SynopsisMetadataItem

- (instancetype) initWithURL:(NSURL *)url
{
    self = [super init];
    if(self)
    {
        self.url = url;
        self.asset = [AVURLAsset URLAssetWithURL:url options:@{AVURLAssetPreferPreciseDurationAndTimingKey : @YES}];
        if(! [self commonLoad] )
            return nil;
            
    }
    
    return self;
}
- (instancetype) initWithAsset:(AVAsset *)asset
{
    self = [super init];
    if(self)
    {
        self.asset = asset;
        if(! [self commonLoad] )
            return nil;
    }
    
    return self;
}

- (BOOL) commonLoad
{
    NSArray* metadataItems = [self.asset metadata];
    
    AVMetadataItem* synopsisMetadataItem = nil;
    
    for(AVMetadataItem* metadataItem in metadataItems)
    {
        if([metadataItem.identifier isEqualToString:kSynopsisMetadataIdentifier])
        {
            synopsisMetadataItem = metadataItem;
            break;
        }
    }
    
    if(synopsisMetadataItem)
    {
        self.decoder = [[SynopsisMetadataDecoder alloc] initWithMetadataItem:synopsisMetadataItem];
        
        self.globalSynopsisMetadata = [self.decoder decodeSynopsisMetadata:synopsisMetadataItem];
        
        return YES;
    }
    
    return NO;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
    return [[SynopsisMetadataItem alloc] initWithURL:self.url];
}


// We test equality based on the file system object we are represeting.

- (BOOL) isEqualToSynopsisMetadataItem:(SynopsisMetadataItem*)object
{
    BOOL equal = [self.url isEqual:object.url];
    
    // helpful for debugging even if stupid
    if(equal)
        return YES;
    
    return NO;

}

- (BOOL) isEqual:(id)object
{
    if(self == object)
        return YES;
    
    return NO;
    
//    if(![object isKindOfClass:[SynopsisMetadataItem class]])
//        return NO;
//    
//    return [self isEqualToSynopsisMetadataItem:(SynopsisMetadataItem*)object];
}

- (NSUInteger) hash
{
    return self.url.hash;
}

- (id) valueForKey:(NSString *)key
{
    NSDictionary* standardDictionary = [self.globalSynopsisMetadata objectForKey:kSynopsisStandardMetadataDictKey];

    if([key isEqualToString:kSynopsisMetadataIdentifier])
        return self.globalSynopsisMetadata;
    
    else if([key isEqualToString:kSynopsisStandardMetadataDictKey])
    {
       return standardDictionary;
    }

    else if(standardDictionary[key])
    {
        return standardDictionary[key];
    }
    else
    {
        return [super valueForKey:key];
    }
}

- (id) valueForUndefinedKey:(NSString *)key
{
    return nil;
}

@end
