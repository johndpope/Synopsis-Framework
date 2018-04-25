//
//  SynopsisCache.m
//  Synopsis-Framework
//
//  Created by vade on 1/15/18.
//  Copyright © 2018 v002. All rights reserved.
//
#import "SynopsisCache.h"
#import <Synopsis/Synopsis.h>
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

@interface SynopsisCache ()
@property (readwrite, strong) NSCache* cache;
@property (readwrite, strong) SynopsisMetadataDecoder* metadataDecoder;
@property (readwrite, strong) NSOperationQueue* cacheMetadataOperationQueue;
@property (readwrite, strong) NSOperationQueue* cacheMediaOperationQueue;
@property (readwrite, atomic, assign) BOOL acceptNewOperations;
@end

@implementation SynopsisCache

+ (instancetype) sharedCache
{
    static SynopsisCache* sharedCache = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedCache = [[SynopsisCache alloc] init];
    });
    
    return sharedCache;
}

- (instancetype) init
{
    self = [super init];
    if(self)
    {
        self.cache = [[NSCache alloc] init];
        self.acceptNewOperations = YES;
        // Metadata decoder isnt strictly thread safe
        // Use a serial queue
        self.metadataDecoder = [[SynopsisMetadataDecoder alloc] initWithVersion:kSynopsisMetadataVersionValue];
        self.cacheMetadataOperationQueue = [[NSOperationQueue alloc] init];
        self.cacheMetadataOperationQueue.maxConcurrentOperationCount = 1;
        self.cacheMetadataOperationQueue.qualityOfService = NSQualityOfServiceBackground;

        self.cacheMediaOperationQueue = [[NSOperationQueue alloc] init];
        self.cacheMediaOperationQueue.maxConcurrentOperationCount = NSOperationQueueDefaultMaxConcurrentOperationCount;
        self.cacheMediaOperationQueue.qualityOfService = NSQualityOfServiceBackground;
    }
    
    return self;
}

- (void) returnOnlyCachedResults
{
    self.acceptNewOperations = NO;
}

- (void) returnCachedAndUncachedResults
{
    self.acceptNewOperations = YES;
}

#pragma mark - Global Metadata

- (NSString*) globalMetadataKeyForItem:(SynopsisMetadataItem* _Nonnull)metadataItem
{
    return [@"GLOBAL-METADATA-" stringByAppendingString:metadataItem.url.absoluteString];
}

- (void) cachedGlobalMetadataForItem:(SynopsisMetadataItem* _Nonnull)metadataItem completionHandler:(SynopsisCacheCompletionHandler)handler
{
    NSBlockOperation* operation = [NSBlockOperation blockOperationWithBlock:^{

        NSDictionary* globalMetadata = nil;
        
        globalMetadata = [self.cache objectForKey:[self globalMetadataKeyForItem:metadataItem]];
        
        //  Generate metadata if we dont have it in the cache
        if(!globalMetadata && self.acceptNewOperations)
        {
            NSArray* metadataItems = metadataItem.asset.metadata;
            for(AVMetadataItem* metadataItem in metadataItems)
            {
                globalMetadata = [self.metadataDecoder decodeSynopsisMetadata:metadataItem];
                if(globalMetadata)
                    break;
            }
            
            // Cache our result for next time
            if(globalMetadata)
                [self.cache setObject:globalMetadata forKey:[self globalMetadataKeyForItem:metadataItem]];
        }
        
        if(handler)
        {
            handler(globalMetadata, nil);
        }
        
    }];
    
    [self.cacheMetadataOperationQueue addOperation:operation];
}

#pragma mark - Image

- (NSString*) imageKeyForItem:(SynopsisMetadataItem* _Nonnull)metadataItem atTime:(CMTime)time
{
    NSString* timeString = (NSString*)CFBridgingRelease(CMTimeCopyDescription(kCFAllocatorDefault, time));
    return [NSString stringWithFormat:@"Image-%@-%@", timeString, metadataItem.url.absoluteString, nil];
}

- (void) cachedImageForItem:(SynopsisMetadataItem* _Nonnull)metadataItem atTime:(CMTime)time completionHandler:(SynopsisCacheCompletionHandler _Nullable )handler;
{
    NSBlockOperation* operation = [NSBlockOperation blockOperationWithBlock:^{

        NSString* key = [self imageKeyForItem:metadataItem atTime:time];

        CGImageRef cachedImage = NULL;
        cachedImage = (CGImageRef) CFBridgingRetain( [self.cache objectForKey:key] );

        if(cachedImage)
        {
            if(handler)
            {
                handler((__bridge id _Nullable)(cachedImage), nil);
            }
        }
        // Generate and cache if nil
        else if(!cachedImage && self.acceptNewOperations)
        {
            AVAssetImageGenerator* imageGenerator = [AVAssetImageGenerator assetImageGeneratorWithAsset:metadataItem.asset];

            imageGenerator.apertureMode = AVAssetImageGeneratorApertureModeCleanAperture;
//            imageGenerator.maximumSize = CGSizeMake(300, 300);
            imageGenerator.appliesPreferredTrackTransform = YES;

            [imageGenerator generateCGImagesAsynchronouslyForTimes:@[ [NSValue valueWithCMTime:kCMTimeZero]] completionHandler:^(CMTime requestedTime, CGImageRef  _Nullable image, CMTime actualTime, AVAssetImageGeneratorResult result, NSError * _Nullable error){

                if(error == nil && image != NULL)
                {
                    [self.cache setObject:(__bridge id _Nonnull)(image) forKey:key];

                    if(handler)
                        handler((__bridge id _Nullable)(cachedImage), nil);

                }
                else
                {
                    NSError* error = [NSError errorWithDomain:NSCocoaErrorDomain code:-1 userInfo:nil];

                    if(handler)
                        handler(nil, error);
                }
            }];
        }
    }];

    [self.cacheMediaOperationQueue addOperation:operation];
}

#pragma mark - Player

- (NSString*) playerKeyForItem:(SynopsisMetadataItem* _Nonnull)metadataItem
{
    return [@"Player-" stringByAppendingString:metadataItem.url.absoluteString];
}


- (void) cachedPlayerForItem:(SynopsisMetadataItem* _Nonnull)metadataItem completionHandler:(SynopsisCacheCompletionHandler _Nullable )handler
{
    NSBlockOperation* operation = [NSBlockOperation blockOperationWithBlock:^{
//
//        NSString* key = [self playerKeyForItem:metadataItem];
//
//        NSImage* cachedPlayer = nil;
//        cachedPlayer = [self.cache objectForKey:key];
//
//        if(cachedPlayer)
//        {
//            if(handler)
//            {
//                handler(cachedPlayer, nil);
//            }
//        }
//        // Generate and cache if nil
//        else if(!cachedPlayer && self.acceptNewOperations)
//        {
//            AVPlayerItem* playerItem = [AVPlayerItem playerItemWithURL:metadataItem.urlAsset];
//            AVPlayer* player = [AVPlayer playerWithURL:playerItem];
//
//            if(player)
//            {
//                [self.cache setObject:player forKey:[self playerKeyForItem:metadataItem]];
//
//                if(handler)
//                    handler(player, nil);
//            }
//            else
//            {
//                if(handler)
//                {
//                    NSError* error = [NSError errorWithDomain:NSCocoaErrorDomain code:-1 userInfo:nil];
//                    handler(nil, error);
//                }
//            }
//        }
    }];
    
    [self.cacheMediaOperationQueue addOperation:operation];

}


@end
