//
//  TensorflowFeatureModule.m
//  Synopsis
//
//  Created by vade on 11/29/16.
//  Copyright © 2016 metavisual. All rights reserved.
//

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

    std::unique_ptr<tensorflow::Session> cinemaNetCoreSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotAnglesSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotFramingSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotSubjectSession;
    std::unique_ptr<tensorflow::Session> cinemaNetShotTypeSession;
    
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
}

@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotAnglesLabels;
@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotFramingLabels;
@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotSubjectLabels;
@property (atomic, readwrite, strong) NSArray<NSString*>* cinemaNetShotTypeLabels;

@property (atomic, readwrite, strong) NSMutableArray* cinemaNetCoreAverageFeatureVector;

@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotAnglesAverageScore;
@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotFramingAverageScore;
@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotSubjectAverageScore;
@property (atomic, readwrite, strong) NSMutableDictionary* cinemaNetShotTypeAverageScore;

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

        NSString* cinemaNetCoreName = @"CinemaNetCore";
        NSString* cinemaNetShotAnglesName = @"CinemaNetShotAnglesClassifier";
        NSString* cinemaNetShotFramingName = @"CinemaNetShotFramingClassifier";
        NSString* cinemaNetShotSubjectName = @"CinemaNetShotSubjectClassifier";
        NSString* cinemaNetShotTypeName = @"CinemaNetShotTypeClassifier";

        cinemaNetCoreInputLayer = "input";
        cinemaNetCoreOutputLayer = "input_1/BottleneckInputPlaceholder";
        cinemaNetClassifierInputLayer = "input_1/BottleneckInputPlaceholder";
        cinemaNetClassifierOutputLayer = "final_result";
        
        self.cinemaNetShotAnglesLabels = @[@"High", @"Tilted", @"Aerial", @"Low"];
        self.cinemaNetShotFramingLabels = @[@"Medium", @"Close Up", @"Extreme Close Up", @"Long", @"Extreme Long"];
        self.cinemaNetShotSubjectLabels = @[@"People", @"Text", @"Face", @"Person", @"Animal", @"Faces"];
        self.cinemaNetShotTypeLabels = @[@"Over The Shoulder", @"Portrait", @"Two Up", @"Master"];

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
        
        // Init Tensorflow Ob
        cinemaNetCoreSession = NULL;
        cinemaNetShotAnglesSession = NULL;
        cinemaNetShotFramingSession = NULL;
        cinemaNetShotSubjectSession = NULL;
        cinemaNetShotTypeSession = NULL;
        
        tensorflow::port::InitMain(NULL, NULL, NULL);
        
#pragma mark - Create TF Graphs
        
        tensorflow::Status load_graph_status;
        
        NSString* cinemaNetCorePath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetCoreName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetCorePath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetCoreGraph);

        // TODO: Modules need better error handling.
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetCore graph");
        }

        NSString* cinemaNetShotAnglePath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotAnglesName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotAnglePath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotAnglesGraph);
        
        // TODO: Modules need better error handling.
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotAngle graph");
        }

        NSString* cinemaNetShotFramingPath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotFramingName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotFramingPath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotFramingGraph);
        
        // TODO: Modules need better error handling.
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotFraming graph");
        }

        NSString* cinemaNetShotSubjectPath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotSubjectName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotSubjectPath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotSubjectGraph);
        
        // TODO: Modules need better error handling.
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotSubject graph");
        }

        NSString* cinemaNetShotTypePath = [[NSBundle bundleForClass:[self class]] pathForResource:cinemaNetShotTypeName ofType:@"pb"];
        load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), [cinemaNetShotTypePath cStringUsingEncoding:NSUTF8StringEncoding], &cinemaNetShotTypeGraph);
        
        // TODO: Modules need better error handling.
        if (!load_graph_status.ok())
        {
            NSLog(@"Unable to load CinemaNetShotType graph");
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

- (SynopsisFrameCacheFormat) currentFrameFormat
{
    return SynopsisFrameCacheFormatOpenCVBGRF32;
}

- (NSDictionary*) analyzedMetadataForCurrentFrame:(matType)frame previousFrame:(matType)lastFrame
{
    self.frameCount++;
    cv::Mat frameMat = frame;

    [self submitAndCacheCurrentVideoCurrentFrame:(matType)frame previousFrame:(matType)lastFrame];
    
    // Actually run the image through the model.
    std::vector<tensorflow::Tensor> cinemaNetCoreOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotAnglesOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotFramingOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotSubjectOutputTensors;
    std::vector<tensorflow::Tensor> cinemaNetShotTypeOutputTensors;

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
            NSLog(@"Error running CinemaNetShotSubject Session");
            return nil;
        }

    }
    
    NSDictionary* labelsAndScores = [self dictionaryFromCoreOutput:cinemaNetCoreOutputTensors
                                                   andAnglesOutput:cinemaNetShotAnglesOutputTensors
                                                    andFrameOutput:cinemaNetShotFramingOutputTensors
                                                  andSubjectOutput:cinemaNetShotSubjectOutputTensors
                                                     andTypeOutput:cinemaNetShotTypeOutputTensors];
    
    return labelsAndScores;
}

- (NSString*) topLabelForScores:(NSMutableDictionary*)scores withThreshhold:(float)thresh
{
    // Average score by number of frames
    for(NSString* key in [scores allKeys])
    {
        NSNumber* score = scores[key];
        NSNumber* newScore = @(score.floatValue / self.frameCount);
        scores[key] = newScore;
    }
    
    NSNumber* topFrameScore = [[[scores allValues] sortedArrayUsingComparator:^NSComparisonResult(id  _Nonnull obj1, id  _Nonnull obj2) {
        
        if([obj1 floatValue] > [obj2 floatValue])
            return NSOrderedAscending;
        else if([obj1 floatValue] < [obj2 floatValue])
            return NSOrderedDescending;
        
        return NSOrderedSame;
    }] firstObject];
    
    NSString* topFrameLabel = nil;
    
    if(topFrameLabel.floatValue >= thresh)
    {
        topFrameLabel = [[scores allKeysForObject:topFrameScore] firstObject];
    }
    
    return topFrameLabel;
}

- (NSDictionary*) finaledAnalysisMetadata
{
    
#if TF_DEBUG_TRACE
    const tensorflow::StepStats& step_stats = run_metadata.step_stats();
    stat_summarizer->ProcessStepStats(step_stats);
    stat_summarizer->PrintStepStats();
#endif
    
    // We only report / include a top score if its over a specific amount
    float topScoreThreshhold = 0.0;
    
    NSString* topAngleLabel = [self topLabelForScores:self.cinemaNetShotAnglesAverageScore withThreshhold:topScoreThreshhold];
    NSString* topFrameLabel = [self topLabelForScores:self.cinemaNetShotFramingAverageScore withThreshhold:topScoreThreshhold];
    NSString* topSubjectLabel = [self topLabelForScores:self.cinemaNetShotSubjectAverageScore withThreshhold:topScoreThreshhold];
    NSString* topTypeLabel = [self topLabelForScores:self.cinemaNetShotTypeAverageScore withThreshhold:topScoreThreshhold];
    
    NSMutableArray* labels = [NSMutableArray array];
    
    if(topAngleLabel)
        [labels addObject:topAngleLabel];
    
    if(topFrameLabel)
        [labels addObject:topFrameLabel];

    if(topSubjectLabel)
        [labels addObject:topSubjectLabel];

    if(topTypeLabel)
        [labels addObject:topTypeLabel];

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
}

#pragma mark - From Old TF Plugin

- (void) submitAndCacheCurrentVideoCurrentFrame:(matType)frame previousFrame:(matType)lastFrame
{
    
#pragma mark - Memory Copy from BGRF32

    // Use OpenCV to normalize input mat

    cv::Mat dst;
    cv::resize(frame, dst, cv::Size(wanted_input_width, wanted_input_height), 0, 0, cv::INTER_LINEAR);

    // Normalize our float input to -1 to 1
    frame = frame - 0.5f;
    frame = frame * 2.0;
    frame = dst;
    
    void* baseAddress = (void*)frame.datastart;
    size_t height = (size_t) frame.rows;
    size_t bytesPerRow =  (size_t) frame.cols * (sizeof(float) * 3); // (BGR)

    auto image_tensor_mapped = resized_tensor.tensor<float, 4>();
    memcpy(image_tensor_mapped.data(), baseAddress, bytesPerRow * height);
    
    dst.release();
}

- (NSDictionary*) dictionaryFromCoreOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetCoreOutputTensors
                           andAnglesOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotAnglesOutputTensors
                            andFrameOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotFramingOutputTensors
                          andSubjectOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotSubjectOutputTensors
                             andTypeOutput:(const std::vector<tensorflow::Tensor>&)cinemaNetShotTypeOutputTensors
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
//            self.cinemaNetCoreAverageFeatureVector[i] = @( MAX(a,b) );
        }
    }
    
    //    NSLog(@"%@", featureElements);
    
#pragma mark - Shot Angles
    
    NSMutableArray* outputAnglesLabels = [NSMutableArray arrayWithCapacity:self.cinemaNetShotAnglesLabels.count];
    NSMutableArray* outputAnglesScores = [NSMutableArray arrayWithCapacity:self.cinemaNetShotAnglesLabels.count];
    
    // 1 = labels and scores
    auto anglepredictions = cinemaNetShotAnglesOutputTensors[0].flat<float>();
    
    for (int index = 0; index < anglepredictions.size(); index += 1)
    {
        const float predictionValue = anglepredictions(index);
        
        NSString* labelKey  = self.cinemaNetShotAnglesLabels[index % anglepredictions.size()];
        
        NSNumber* currentLabelScore = self.cinemaNetShotAnglesAverageScore[labelKey];
        
        NSNumber* incrementedScore = @([currentLabelScore floatValue] + predictionValue );
        self.cinemaNetShotAnglesAverageScore[labelKey] = incrementedScore;
        
        [outputAnglesLabels addObject:labelKey];
        [outputAnglesScores addObject:@(predictionValue)];
    }
    

#pragma mark - Shot Framing
    
    NSMutableArray* outputFramingLabels = [NSMutableArray arrayWithCapacity:self.cinemaNetShotFramingLabels.count];
    NSMutableArray* outputFramingScores = [NSMutableArray arrayWithCapacity:self.cinemaNetShotFramingLabels.count];

    // 1 = labels and scores
    auto framepredictions = cinemaNetShotFramingOutputTensors[0].flat<float>();

    for (int index = 0; index < framepredictions.size(); index += 1)
    {
        const float predictionValue = framepredictions(index);

        NSString* labelKey  = self.cinemaNetShotFramingLabels[index % framepredictions.size()];

        NSNumber* currentLabelScore = self.cinemaNetShotFramingAverageScore[labelKey];

        NSNumber* incrementedScore = @([currentLabelScore floatValue] + predictionValue );
        self.cinemaNetShotFramingAverageScore[labelKey] = incrementedScore;

        [outputFramingLabels addObject:labelKey];
        [outputFramingScores addObject:@(predictionValue)];
    }

#pragma mark - Shot Subject
    
    NSMutableArray* outputSubjectLabels = [NSMutableArray arrayWithCapacity:self.cinemaNetShotSubjectLabels.count];
    NSMutableArray* outputSubjectScores = [NSMutableArray arrayWithCapacity:self.cinemaNetShotSubjectLabels.count];
    
    // 1 = labels and scores
    auto subjectpredictions = cinemaNetShotSubjectOutputTensors[0].flat<float>();
    
    for (int index = 0; index < subjectpredictions.size(); index += 1)
    {
        const float predictionValue = subjectpredictions(index);
        
        NSString* labelKey  = self.cinemaNetShotSubjectLabels[index % subjectpredictions.size()];
        
        NSNumber* currentLabelScore = self.cinemaNetShotSubjectAverageScore[labelKey];
        
        NSNumber* incrementedScore = @([currentLabelScore floatValue] + predictionValue );
        self.cinemaNetShotSubjectAverageScore[labelKey] = incrementedScore;
        
        [outputSubjectLabels addObject:labelKey];
        [outputSubjectScores addObject:@(predictionValue)];
    }
    
#pragma mark - Shot Type
    
    NSMutableArray* outputTypeLabels = [NSMutableArray arrayWithCapacity:self.cinemaNetShotTypeLabels.count];
    NSMutableArray* outputTypeScores = [NSMutableArray arrayWithCapacity:self.cinemaNetShotTypeLabels.count];
    
    // 1 = labels and scores
    auto typepredictions = cinemaNetShotTypeOutputTensors[0].flat<float>();
    
    for (int index = 0; index < typepredictions.size(); index += 1)
    {
        const float predictionValue = typepredictions(index);
        
        NSString* labelKey  = self.cinemaNetShotTypeLabels[index % typepredictions.size()];
        
        NSNumber* currentLabelScore = self.cinemaNetShotTypeAverageScore[labelKey];
        
        NSNumber* incrementedScore = @([currentLabelScore floatValue] + predictionValue );
        self.cinemaNetShotTypeAverageScore[labelKey] = incrementedScore;
        
        [outputTypeLabels addObject:labelKey];
        [outputTypeScores addObject:@(predictionValue)];
    }

//    NSLog(@"%@, %@", outputFramingLabels, outputFramingScores);
//
//    NSLog(@"%@, %@", outputSubjectLabels, outputSubjectScores);
//
//    NSLog(@"%@, %@", outputTypeLabels, outputTypeScores);
    
#pragma mark - Fin
    
    return @{
             kSynopsisStandardMetadataFeatureVectorDictKey : featureElements ,
//             @"Labels" : outputLabels,
//             @"Scores" : outputScores,
            };
}

@end
