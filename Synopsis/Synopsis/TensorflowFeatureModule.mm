//
//  TensorflowFeatureModule.m
//  Synopsis
//
//  Created by vade on 11/29/16.
//  Copyright © 2016 metavisual. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import "SynopsisVideoFrameOpenCV.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wconversion"

#import "tensorflow/cc/ops/const_op.h"
#import "tensorflow/cc/ops/image_ops.h"
#import "tensorflow/cc/ops/standard_ops.h"
#import "tensorflow/core/framework/graph.pb.h"
#import "tensorflow/core/framework/tensor.h"
#import "tensorflow/core/graph/default_device.h"
#import "tensorflow/core/graph/graph_def_builder.h"
#import "tensorflow/core/lib/core/errors.h"
#import "tensorflow/core/lib/core/stringpiece.h"
#import "tensorflow/core/lib/core/threadpool.h"
#import "tensorflow/core/lib/io/path.h"
#import "tensorflow/core/lib/strings/stringprintf.h"
#import "tensorflow/core/platform/init_main.h"
#import "tensorflow/core/platform/logging.h"
#import "tensorflow/core/platform/types.h"
#import "tensorflow/core/public/session.h"
#import "tensorflow/core/util/command_line_flags.h"
#import "tensorflow/core/util/stat_summarizer.h"
#import "tensorflow/core/util/tensor_format.h"

#import "TensorflowFeatureModule.h"

#import <fstream>
#import <vector>

#pragma GCC diagnostic pop

#define TF_DEBUG_TRACE 0

@interface TensorflowFeatureModule ()
{
    // CinemaNet consists of a core graph
    // which creates feature vectors - this does the heavy work
    // and multiple classifiers we run independently.
    tensorflow::GraphDef cinemaNetCoreGraph;
    tensorflow::GraphDef cinemaNetShotAnglesGraph;
    tensorflow::GraphDef cinemaNetShotFramingGraph;
    tensorflow::GraphDef cinemaNetShotSubjectGraph;
    tensorflow::GraphDef cinemaNetShotTypeGraph;

    tensorflow::GraphDef imageNetGraph;
    tensorflow::GraphDef placesNetGraph;

    std::unique_ptr<tensorflow::Session> cinemaNetCoreSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotAnglesSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotFramingSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotSubjectSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotTypeSession;

    std::unique_ptr<tensorflow::Session> imageNetSession;
    std::unique_ptr<tensorflow::Session> placesNetSession;

#if TF_DEBUG_TRACE
    std::unique_ptr<tensorflow::StatSummarizer> stat_summarizer;
    tensorflow::RunMetadata run_metadata;
#endif
    
    // Cached resized tensor from our input buffer (image)
    tensorflow::Tensor resized_tensor;
    
    // Core input and output tensors to generate feature vectors
    std::string cinemaNetCoreInputLayer;
    std::string cinemaNetCoreOutputLayer;

    std::string cinemaNetClassifierInputLayer;
    std::string cinemaNetClassifierOutputLayer;

    // Cant quite figure out how to normaize the ImageNet model input
    // So it matches transfer learning models (Places, CinemaNet* etc)
    // (pretrained vs fine tuned networks have diff labels for input / output!?)
    std::string imageNetClassifierInputLayer;
    std::string imageNetClassifierOutputLayer;
}

@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotAnglesLabels;
@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotFramingLabels;
@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotSubjectLabels;
@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotTypeLabels;

@property (atomic, readwrite, strong) NSArray<NSString*>* imageNetLabels;
@property (atomic, readwrite, strong) NSArray<NSString*>* placesNetLabels;

@property (atomic, readwrite, strong) NSMutableArray* cinemaNetCoreAverageFeatureVector;

@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotAnglesAverageScore;
@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotFramingAverageScore;
@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotSubjectAverageScore;
@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotTypeAverageScore;

@property (atomic, readwrite, strong) NSMutableDictionary* imageNetAverageScore;
@property (atomic, readwrite, strong) NSMutableDictionary* placesNetAverageScore;

@property (atomic, readwrite, assign) NSUInteger frameCount;

@end

#define wanted_input_width 224
#define wanted_input_height 224
#define wanted_input_channels 3

@implementation TensorflowFeatureModule

- (instancetype) initWithQualityHint:(SynopsisAnalysisQualityHint)qualityHint
{
    self = [super initWithQualityHint:qualityHint];
    if(self)
    {
        self.cinemaNetShotAnglesAverageScore = [NSMutableDictionary dictionary];
        self.cinemaNetShotFramingAverageScore = [NSMutableDictionary dictionary];
        self.cinemaNetShotSubjectAverageScore = [NSMutableDictionary dictionary];
        self.cinemaNetShotTypeAverageScore = [NSMutableDictionary dictionary];
        self.imageNetAverageScore = [NSMutableDictionary dictionary];
        self.placesNetAverageScore = [NSMutableDictionary dictionary];
        
        cinemaNetCoreInputLayer = "input";
        cinemaNetCoreOutputLayer = "input_1/BottleneckInputPlaceholder";
        cinemaNetClassifierInputLayer = "input_1/BottleneckInputPlaceholder";
        cinemaNetClassifierOutputLayer = "final_result";
        
        // Cant quite figure out how to normaize the ImageNet model input
        // So it matches transfer learning models (Places, CinemaNet* etc)
        // (pretrained vs fine tuned networks have diff labels for input / output!?)
        imageNetClassifierInputLayer = "MobilenetV1/Predictions/Reshape";
        imageNetClassifierOutputLayer = "MobilenetV1/Predictions/Reshape_1";
        
        self.cinemaNetShotAnglesLabels = [self labelArrayFromLabelFileName:@"CinemaNetShotAnglesLabels"];
        self.cinemaNetShotFramingLabels = [self labelArrayFromLabelFileName:@"CinemaNetShotFramingLabels"];
        self.cinemaNetShotSubjectLabels = [self labelArrayFromLabelFileName:@"CinemaNetShotSubjectLabels"];
        self.cinemaNetShotTypeLabels = [self labelArrayFromLabelFileName:@"CinemaNetShotTypeLabels"];

        self.imageNetLabels = [self labelArrayFromLabelFileName:@"ImageNetLabels"];
        self.placesNetLabels = [self labelArrayFromLabelFileName:@"PlacesNetLabels"];

        self.cinemaNetCoreAverageFeatureVector = nil;

        for(NSString* label in self.cinemaNetShotAnglesLabels)
        {
            self.cinemaNetShotAnglesAverageScore[label] = @(0.0);
        }

        for(NSString* label in self.cinemaNetShotFramingLabels)
        {
            self.cinemaNetShotFramingAverageScore[label] = @(0.0);
        }

        for(NSString* label in self.cinemaNetShotSubjectLabels)
        {
            self.cinemaNetShotSubjectAverageScore[label] = @(0.0);
        }

        for(NSString* label in self.cinemaNetShotTypeLabels)
        {
            self.cinemaNetShotTypeAverageScore[label] = @(0.0);
        }

        for(NSString* label in self.imageNetLabels)
        {
            self.imageNetAverageScore[label] = @(0.0);
        }

        for(NSString* label in self.placesNetLabels)
        {
            self.placesNetAverageScore[label] = @(0.0);
        }
        
        // Init Tensorflow Ob
        cinemaNetCoreSession = NULL;
        cinemaNetShotAnglesSession = NULL;
        cinemaNetShotFramingSession = NULL;
        cinemaNetShotSubjectSession = NULL;
        cinemaNetShotTypeSession = NULL;
        
        tensorflow::port::InitMain(NULL, NULL, NULL);
        
#pragma mark - Create TF Graphs
        
        tensorflow::Status load_graph_status;
        
        NSString* cinemaNetCoreName = @"CinemaNetCore";
        NSString* cinemaNetCorePath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetCoreName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetCorePath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetCoreGraph);

        // TODO: Modules need better error handling.
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetCore graph");
        }

        NSString* cinemaNetShotAnglesName = @"CinemaNetShotAnglesClassifier";
        NSString* cinemaNetShotAnglePath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotAnglesName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotAnglePath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotAnglesGraph);
        
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotAngle graph");
        }

        NSString* cinemaNetShotFramingName = @"CinemaNetShotFramingClassifier";
        NSString* cinemaNetShotFramingPath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotFramingName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotFramingPath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotFramingGraph);
        
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotFraming graph");
        }

        NSString* cinemaNetShotSubjectName = @"CinemaNetShotSubjectClassifier";
        NSString* cinemaNetShotSubjectPath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotSubjectName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotSubjectPath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotSubjectGraph);
        
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotSubject graph");
        }

        NSString* cinemaNetShotTypeName = @"CinemaNetShotTypeClassifier";
        NSString* cinemaNetShotTypePath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotTypeName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotTypePath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotTypeGraph);
        
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotType graph");
        }

        NSString* imageNetName = @"ImageNetClassifier";
        NSString* imageNetPath = [[NSBundle bundleForClass:[self class]] pathForResource:imageNetName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [imageNetPath cStringUsingEncoding:NSUTF8StringEncoding], &imageNetGraph);
        
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load ImageNet graph");
        }

        NSString* placesNetName = @"PlacesNetClassifier";
        NSString* placesNetPath = [[NSBundle bundleForClass:[self class]] pathForResource:placesNetName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [placesNetPath cStringUsingEncoding:NSUTF8StringEncoding], &placesNetGraph);
        
        // TODO: Modules need better error handling.
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load ImageNet graph");
        }

#pragma mark - Create TF Sessions
        
        tensorflow::SessionOptions options;
        tensorflow::Status session_create_status;
        
        cinemaNetCoreSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
        session_create_status = cinemaNetCoreSession->Create(cinemaNetCoreGraph);
        if (!session_create_status.ok())
        {
            NSLog(@"Unable to create CinemaNetCore session");
        }
        
        cinemaNetShotAnglesSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
        session_create_status = cinemaNetShotAnglesSession->Create(cinemaNetShotAnglesGraph);
        if (!session_create_status.ok())
        {
            NSLog(@"Unable to create CinemaNetShotAngles session");
        }
        
        cinemaNetShotFramingSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
        session_create_status = cinemaNetShotFramingSession->Create(cinemaNetShotFramingGraph);
        if (!session_create_status.ok())
        {
            NSLog(@"Unable to create CinemaNetShotFraming session");
        }

        cinemaNetShotSubjectSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
        session_create_status = cinemaNetShotSubjectSession->Create(cinemaNetShotSubjectGraph);
        if (!session_create_status.ok())
        {
            NSLog(@"Unable to create CinemaNetShotSubject session");
        }

        cinemaNetShotTypeSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
        session_create_status = cinemaNetShotTypeSession->Create(cinemaNetShotTypeGraph);
        if (!session_create_status.ok())
        {
            NSLog(@"Unable to create CinemaNetShotType session");
        }

        imageNetSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
        session_create_status = imageNetSession->Create(imageNetGraph);
        if (!session_create_status.ok())
        {
            NSLog(@"Unable to create ImageNet session");
        }

        placesNetSession = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
        session_create_status = placesNetSession->Create(placesNetGraph);
        if (!session_create_status.ok())
        {
            NSLog(@"Unable to create PlacesNet session");
        }

        
#pragma mark Create TF Requirements
        
        tensorflow::TensorShape shape = tensorflow::TensorShape({1, wanted_input_height, wanted_input_width, wanted_input_channels});
        resized_tensor = tensorflow::Tensor( tensorflow::DT_FLOAT, shape );
        

#if TF_DEBUG_TRACE
        stat_summarizer = std::unique_ptr<tensorflow::StatSummarizer>(new tensorflow::StatSummarizer(inceptionGraphDef));
#endif

    }
    return self;
}

- (void) dealloc
{
}

- (NSString*) moduleName
{
    return kSynopsisStandardMetadataFeatureVectorDictKey;//@"Feature";
}

+ (SynopsisVideoBacking) requiredVideoBacking
{
    return SynopsisVideoBackingCPU;
}

+ (SynopsisVideoFormat) requiredVideoFormat
{
    return SynopsisVideoFormatBGRF32;
}

- (NSArray<NSString*>*)labelArrayFromLabelFileName:(NSString*)labelName
{
    NSString* labelPath = [[NSBundle bundleForClass:[self class]] pathForResource:labelName ofType:@"txt"];

    NSString* rawLabels = [NSString stringWithContentsOfURL:[NSURL fileURLWithPath:labelPath] usedEncoding:NULL error:nil];
    
    rawLabels = [rawLabels capitalizedString];
    
    NSArray<NSString *> * labels = [rawLabels componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    
    NSMutableArray<NSString *> * mutableLabels = [labels mutableCopy];
    
    NSMutableIndexSet* indicesToRemove = [[NSMutableIndexSet alloc] init];
    [mutableLabels enumerateObjectsUsingBlock:^(NSString * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        if(obj.length == 0)
        {
            [indicesToRemove addIndex:idx];
        }
        if([obj isEqualToString:@" "])
        {
            [indicesToRemove addIndex:idx];
        }
    }];
    
    [mutableLabels removeObjectsAtIndexes:indicesToRemove];
    
    return [mutableLabels copy];
}

- (NSDictionary*) analyzedMetadataForCurrentFrame:(id<SynopsisVideoFrame>)frame previousFrame:(id<SynopsisVideoFrame>)lastFrame;
{
    SynopsisVideoFrameOpenCV* frameCV = (SynopsisVideoFrameOpenCV*)frame;
    SynopsisVideoFrameOpenCV* previousFrameCV = (SynopsisVideoFrameOpenCV*)lastFrame;
    
    self.frameCount++;

    [self submitAndCacheCurrentVideoCurrentFrame:frameCV.mat previousFrame:previousFrameCV.mat];

    // Actually run the image through the model.
    std::vector<tensorflow::Tensor> cinemaNetCoreOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotAnglesOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotFramingOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotSubjectOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotTypeOutputTensors;

    std::vector<tensorflow::Tensor> imageNetOutputTensors;
    std::vector<tensorflow::Tensor> placesNetOutputTensors;

#if TF_DEBUG_TRACE
    tensorflow::RunOptions run_options;
    run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
    tensorflow::Status run_status = cinemaNetCoreSession->Run(run_options, { {cinemaNetCoreInputLayer, resized_tensor} }, {cinemaNetCoreOutputLayer}, {}, &cinemaNetCoreOutputTensors, &run_metadata);
#else
    tensorflow::Status run_status = cinemaNetCoreSession->Run({ {cinemaNetCoreInputLayer, resized_tensor} }, {cinemaNetCoreOutputLayer}, {}, &cinemaNetCoreOutputTensors);
#endif

    if (!run_status.ok())
    {
        NSLog(@"Error running CinemaNetCore Session");
        return nil;
    }

    if(!cinemaNetCoreOutputTensors.empty())
    {
        run_status = cinemaNetShotAnglesSession->Run({ {cinemaNetClassifierInputLayer, cinemaNetCoreOutputTensors[0]} }, {cinemaNetClassifierOutputLayer}, {}, &cinemaNetShotAnglesOutputTensors);
        
        if (!run_status.ok())
        {
            NSLog(@"Error running CinemaNetShotAngles Session");
            return nil;
        }
        
        run_status = cinemaNetShotFramingSession->Run({ {cinemaNetClassifierInputLayer, cinemaNetCoreOutputTensors[0]} }, {cinemaNetClassifierOutputLayer}, {}, &cinemaNetShotFramingOutputTensors);
        
        if (!run_status.ok())
        {
            NSLog(@"Error running CinemaNetShotFraming Session");
            return nil;
        }
        
        run_status = cinemaNetShotSubjectSession->Run({ {cinemaNetClassifierInputLayer, cinemaNetCoreOutputTensors[0]} }, {cinemaNetClassifierOutputLayer}, {}, &cinemaNetShotSubjectOutputTensors);
        
        if (!run_status.ok())
        {
            NSLog(@"Error running CinemaNetShotSubject Session");
            return nil;
        }
        
        run_status = cinemaNetShotTypeSession->Run({ {cinemaNetClassifierInputLayer, cinemaNetCoreOutputTensors[0]} }, {cinemaNetClassifierOutputLayer}, {}, &cinemaNetShotTypeOutputTensors);
        
        if (!run_status.ok())
        {
            NSLog(@"Error running CinemaNetShotType Session");
            return nil;
        }

        run_status = imageNetSession->Run({ {imageNetClassifierInputLayer, cinemaNetCoreOutputTensors[0]} }, {imageNetClassifierOutputLayer}, {}, &imageNetOutputTensors);

        if (!run_status.ok())
        {
            NSLog(@"Error running ImageNet Session");
            return nil;
        }
    
        run_status = placesNetSession->Run({ {cinemaNetClassifierInputLayer, cinemaNetCoreOutputTensors[0]} }, {cinemaNetClassifierOutputLayer}, {}, &placesNetOutputTensors);
        
        if (!run_status.ok())
        {
            NSLog(@"Error running PlacesNet Session");
            return nil;
        }

    }
    
    NSDictionary* labelsAndScores = [self dictionaryFromCoreOutput:cinemaNetCoreOutputTensors
                                                   andAnglesOutput:cinemaNetShotAnglesOutputTensors
                                                    andFrameOutput:cinemaNetShotFramingOutputTensors
                                                  andSubjectOutput:cinemaNetShotSubjectOutputTensors
                                                     andTypeOutput:cinemaNetShotTypeOutputTensors
                                                 andImageNetOutput:imageNetOutputTensors
                                                andPlacesNetOutput:placesNetOutputTensors
                                     ];
    
    return labelsAndScores;
}

- (NSArray<NSString*>*) topLabelForScores:(NSDictionary*)scoresDict withThreshhold:(float)thresh maxLabelCount:(NSUInteger)maxLabels
{
    NSMutableDictionary* scores = [scoresDict mutableCopy];
        
    NSArray* sortedScores = [[scores allValues] sortedArrayUsingComparator:^NSComparisonResult(id  _Nonnull obj1, id  _Nonnull obj2) {
        
        if([obj1 floatValue] > [obj2 floatValue])
            return NSOrderedAscending;
        else if([obj1 floatValue] < [obj2 floatValue])
            return NSOrderedDescending;
        
        return NSOrderedSame;
    }] ;
    
    //    NSString* topFrameLabel = nil;
    NSMutableArray* top3 = [NSMutableArray array];
    // Modulate percentage based off of number of possible categories?
    [sortedScores enumerateObjectsUsingBlock:^(NSNumber*  _Nonnull score, NSUInteger idx, BOOL * _Nonnull stop) {
        if(idx >= (maxLabels - 1) )
            *stop = YES;
        
        if(score.floatValue >= (thresh / scores.allKeys.count))
        {
            NSString* scoreLabel = [[scores allKeysForObject:score] firstObject];
            [top3 addObject:scoreLabel];
        }
    }];
    
    
    return top3;
}

- (NSDictionary*) finaledAnalysisMetadata
{
    
#if TF_DEBUG_TRACE
    const tensorflow::StepStats& step_stats = run_metadata.step_stats();
    stat_summarizer->ProcessStepStats(step_stats);
    stat_summarizer->PrintStepStats();
#endif
    
    // We only report / include a top score if its over a specific amount
    // Note this threshold is modulated by the number of labels
    // See topLabelForScores:withThreshhold:maxLabelCount:
    float topScoreThreshhold = 0.85;
    
    NSArray<NSString*>* topAngleLabel = [self topLabelForScores:self.cinemaNetShotAnglesAverageScore withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* topFrameLabel = [self topLabelForScores:self.cinemaNetShotFramingAverageScore withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* topSubjectLabel = [self topLabelForScores:self.cinemaNetShotSubjectAverageScore withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* topTypeLabel = [self topLabelForScores:self.cinemaNetShotTypeAverageScore withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* imageNetLabel = [self topLabelForScores:self.imageNetAverageScore withThreshhold:topScoreThreshhold maxLabelCount:2];
    NSArray<NSString*>* placesNetLabel = [self topLabelForScores:self.placesNetAverageScore withThreshhold:topScoreThreshhold maxLabelCount:3];

    NSMutableArray* labels = [NSMutableArray array];
    
    if(topAngleLabel)
    {
        [labels addObject:@"Shot Angle:"];
        [labels addObjectsFromArray:topAngleLabel];
    }
    if(topFrameLabel)
    {
        [labels addObject:@"Shot Framing:"];
        [labels addObjectsFromArray:topFrameLabel];
    }
    if(topSubjectLabel)
    {
        [labels addObject:@"Shot Subject:"];
        [labels addObjectsFromArray:topSubjectLabel];
    }
    if(topTypeLabel)
    {
        [labels addObject:@"Shot Type:"];
        [labels addObjectsFromArray:topTypeLabel];
    }
    if(imageNetLabel)
    {
        [labels addObject:@"Objects:"];
        [labels addObjectsFromArray:imageNetLabel];
    }
    if(placesNetLabel)
    {
        [labels addObject:@"Location:"];
        [labels addObjectsFromArray:placesNetLabel];
    }

    [self shutdownTF];

    return @{
             kSynopsisStandardMetadataFeatureVectorDictKey : self.cinemaNetCoreAverageFeatureVector,
                 kSynopsisStandardMetadataDescriptionDictKey : [labels copy],
//             kSynopsisStandardMetadataLabelsDictKey : [self.averageLabelScores allKeys],
//             kSynopsisStandardMetadataScoreDictKey : [self.averageLabelScores allValues],
            };
}

- (void) shutdownTF
{
    if(cinemaNetCoreSession != NULL)
    {
        tensorflow::Status close_graph_status = cinemaNetCoreSession->Close();
        if (!close_graph_status.ok())
        {
            NSLog(@"Error Closing Session");
        }
    }

    if(cinemaNetShotAnglesSession != NULL)
    {
        tensorflow::Status close_graph_status = cinemaNetShotAnglesSession->Close();
        if (!close_graph_status.ok())
        {
            NSLog(@"Error Closing Session");
        }
    }
    
    if(cinemaNetShotFramingSession != NULL)
    {
        tensorflow::Status close_graph_status = cinemaNetShotFramingSession->Close();
        if (!close_graph_status.ok())
        {
            NSLog(@"Error Closing Session");
        }
    }

    if(cinemaNetShotSubjectSession != NULL)
    {
        tensorflow::Status close_graph_status = cinemaNetShotSubjectSession->Close();
        if (!close_graph_status.ok())
        {
            NSLog(@"Error Closing Session");
        }
    }

    if(cinemaNetShotTypeSession != NULL)
    {
        tensorflow::Status close_graph_status = cinemaNetShotTypeSession->Close();
        if (!close_graph_status.ok())
        {
            NSLog(@"Error Closing Session");
        }
    }

    if(imageNetSession != NULL)
    {
        tensorflow::Status close_graph_status = imageNetSession->Close();
        if (!close_graph_status.ok())
        {
            NSLog(@"Error Closing Session");
        }
    }

    if(placesNetSession != NULL)
    {
        tensorflow::Status close_graph_status = placesNetSession->Close();
        if (!close_graph_status.ok())
        {
            NSLog(@"Error Closing Session");
        }
    }

}

#pragma mark - From Old TF Plugin

- (void) submitAndCacheCurrentVideoCurrentFrame:(cv::Mat)frame previousFrame:(cv::Mat)lastFrame
{
    
#pragma mark - Memory Copy from BGRF32

    // Use OpenCV to normalize input mat

    cv::Mat dst;
    cv::resize(frame, dst, cv::Size(wanted_input_width, wanted_input_height), 0, 0, cv::INTER_CUBIC);

    // I weirdly get more accurate (?) results if I dont normalize?
    // Is OpenCV doing something behind my back?
    // Normalize our float input to -1 to 1
//    dst = dst - 0.5f;
//    dst = dst * 2.0;
    
    const float* baseAddress = (const float*)dst.datastart;
    size_t height = (size_t) dst.rows;
    size_t width =  (size_t) dst.cols;
    size_t depth = 3;
//    size_t bytesPerRow =  (size_t) width * (sizeof(float) * depth); // (BGR)

    auto image_tensor_mapped = resized_tensor.tensor<float, 4>();
    
    // This isnt quite right, we dont map the expected layout for TF as NHWC
    // Rather its NWHC I believe?
    // In theory we can make an empty tensor and point it to the base address directly
    // For zero copy?
    // memcpy(image_tensor_mapped.data(), baseAddress, bytesPerRow * height);

    // Maybe theres a good way of handling that in OpenCV to optimize this out?
    for (int y = 0; y < height; ++y)
    {
        const float* source_row = baseAddress + (y * width * depth);
        for (int x = 0; x < width; ++x)
        {
            const float* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c)
            {
                const float* source_value = source_pixel + c;

                image_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }

    dst.release();
}

- (NSDictionary*) predictionFromTensor:(const std::vector<tensorflow::Tensor>&)tensor withLabels:(NSArray<NSString*>*)labels
{
    NSMutableDictionary* labelAndScoreDict = [NSMutableDictionary new];
    
    // 1 = labels and scores
    auto predictions = tensor[0].flat<float>();
    
    for (int index = 0; index < predictions.size(); index += 1)
    {
        const float predictionValue = predictions(index);
        
        NSString* labelKey  = labels[index % predictions.size()];
        
        NSNumber* score = @(predictionValue );
        
        labelAndScoreDict[labelKey] = score;
    }

    return labelAndScoreDict;
}

- (NSMutableDictionary*) averageLabelsAndScores:(NSDictionary*)averageLabelsScores withNewLabelsAndScores:(NSDictionary*)newLabelsAndScores
{
    NSMutableDictionary* newAverageLabelsAndScores = [averageLabelsScores mutableCopy];
    
    [averageLabelsScores enumerateKeysAndObjectsUsingBlock:^(NSString* _Nonnull label, NSNumber*  _Nonnull score, BOOL * _Nonnull stop) {
        
        if(newLabelsAndScores[label])
        {
            NSNumber* newScore = newLabelsAndScores[label];
            
            NSNumber* newAverageScore = @( (newScore.floatValue + score.floatValue) * 0.5 );
            
            newAverageLabelsAndScores[label] = newAverageScore;
        }
    }];
    
    return newAverageLabelsAndScores;
}

- (NSDictionary*) dictionaryFromCoreOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetCoreOutputTensors
                           andAnglesOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotAnglesOutputTensors
                            andFrameOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotFramingOutputTensors
                          andSubjectOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotSubjectOutputTensors
                             andTypeOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotTypeOutputTensors
                         andImageNetOutput:(const std::vector<tensorflow::Tensor>&)imageNetOutputTensors
                        andPlacesNetOutput:(const std::vector<tensorflow::Tensor>&)placesNetOutputTensors
{
    
#pragma mark - Feature Vector
    
    // 0 is feature vector
    tensorflow::Tensor feature = cinemaNetCoreOutputTensors[0];
    int64_t numElements = feature.NumElements();
    tensorflow::TTypes<float>::Flat featureVec = feature.flat<float>();
    
    NSMutableArray* featureElements = [NSMutableArray arrayWithCapacity:numElements];
    
    for(int i = 0; i < numElements; i++)
    {
        if( ! std::isnan(featureVec(i)))
        {
            [featureElements addObject:@( featureVec(i) ) ];
        }
        else
        {
            NSLog(@"Feature is Nan");
        }
    }
    
    if(self.cinemaNetCoreAverageFeatureVector == nil)
    {
        self.cinemaNetCoreAverageFeatureVector = featureElements;
    }
    else
    {
        // average each vector element with the prior
        for(int i = 0; i < featureElements.count; i++)
        {
            float  a = [featureElements[i] floatValue];
            float  b = [self.cinemaNetCoreAverageFeatureVector[i] floatValue];
            
            self.cinemaNetCoreAverageFeatureVector[i] = @( (a + b) * 0.5 );
        }
    }
    
#pragma mark - Shot Angles
    
    NSDictionary* angleLabelsAndScores = [self predictionFromTensor:cinemaNetShotAnglesOutputTensors withLabels:self.cinemaNetShotAnglesLabels];
    self.cinemaNetShotAnglesAverageScore = [self averageLabelsAndScores:self.cinemaNetShotAnglesAverageScore withNewLabelsAndScores:angleLabelsAndScores];
    
#pragma mark - Shot Framing

    NSDictionary* framingLabelsAndScores = [self predictionFromTensor:cinemaNetShotFramingOutputTensors withLabels:self.cinemaNetShotFramingLabels];
    self.cinemaNetShotFramingAverageScore = [self averageLabelsAndScores:self.cinemaNetShotFramingAverageScore withNewLabelsAndScores:framingLabelsAndScores];

#pragma mark - Shot Subject
    
    NSDictionary* subjectLabelsAndScores = [self predictionFromTensor:cinemaNetShotSubjectOutputTensors withLabels:self.cinemaNetShotSubjectLabels];
    self.cinemaNetShotSubjectAverageScore = [self averageLabelsAndScores:self.cinemaNetShotSubjectAverageScore withNewLabelsAndScores:subjectLabelsAndScores];

#pragma mark - Shot Type
    
    NSDictionary* typeLabelsAndScores = [self predictionFromTensor:cinemaNetShotTypeOutputTensors withLabels:self.cinemaNetShotTypeLabels];
    self.cinemaNetShotTypeAverageScore = [self averageLabelsAndScores:self.cinemaNetShotTypeAverageScore withNewLabelsAndScores:typeLabelsAndScores];
    
#pragma mark - ImageNet
    
    NSDictionary* imageLabelsAndScores = [self predictionFromTensor:imageNetOutputTensors withLabels:self.imageNetLabels];
    self.imageNetAverageScore = [self averageLabelsAndScores:self.imageNetAverageScore withNewLabelsAndScores:imageLabelsAndScores];

#pragma mark - PlacesNet
    
    NSDictionary* placesLabelsAndScores = [self predictionFromTensor:placesNetOutputTensors withLabels:self.placesNetLabels];
    self.placesNetAverageScore = [self averageLabelsAndScores:self.placesNetAverageScore withNewLabelsAndScores:placesLabelsAndScores];

#pragma mark -
    
    float topScoreThreshhold = FLT_MIN;
    
    NSArray<NSString*>* topAngleLabel = [self topLabelForScores:angleLabelsAndScores withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* topFrameLabel = [self topLabelForScores:framingLabelsAndScores withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* topSubjectLabel = [self topLabelForScores:subjectLabelsAndScores withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* topTypeLabel = [self topLabelForScores:typeLabelsAndScores withThreshhold:topScoreThreshhold maxLabelCount:1];
    NSArray<NSString*>* imageNetLabel = [self topLabelForScores:imageLabelsAndScores withThreshhold:topScoreThreshhold maxLabelCount:2];
    NSArray<NSString*>* placesNetLabel = [self topLabelForScores:placesLabelsAndScores withThreshhold:topScoreThreshhold maxLabelCount:3];
    
    NSMutableArray* labels = [NSMutableArray array];
    
    if(topAngleLabel)
    {
        [labels addObject:@"Shot Angle:"];
        [labels addObjectsFromArray:topAngleLabel];
    }
    if(topFrameLabel)
    {
        [labels addObject:@"Shot Framing:"];
        [labels addObjectsFromArray:topFrameLabel];
    }
    if(topSubjectLabel)
    {
        [labels addObject:@"Shot Subject:"];
        [labels addObjectsFromArray:topSubjectLabel];
    }
    if(topTypeLabel)
    {
        [labels addObject:@"Shot Type:"];
        [labels addObjectsFromArray:topTypeLabel];
    }
    if(imageNetLabel)
    {
        [labels addObject:@"Objects:"];
        [labels addObjectsFromArray:imageNetLabel];
    }
    if(placesNetLabel)
    {
        [labels addObject:@"Location:"];
        [labels addObjectsFromArray:placesNetLabel];
    }
#pragma mark - Fin
    
    return @{
             kSynopsisStandardMetadataFeatureVectorDictKey : featureElements ,
             kSynopsisStandardMetadataDescriptionDictKey : labels,
 //             kSynopsisStandardMetadataLabelsDictKey : @[outputAnglesLabels,outputFramingLabels, outputSubjectLabels, outputTypeLabels, outputPlacesNetLabels],
//             kSynopsisStandardMetadataScoreDictKey : @[outputAnglesScores,outputFramingScores, outputSubjectScores, outputTypeScores, outputPlacesNetScores],
            };
}

@end
