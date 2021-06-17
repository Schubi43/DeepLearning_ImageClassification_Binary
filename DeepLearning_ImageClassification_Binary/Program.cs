using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

namespace DeepLearning_ImageClassification_Binary
{
    class Program
    {
        static void Main(string[] args)
        {
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var assetsRelativePath = Path.Combine(projectDirectory, "assets");
            MLContext mLContext = new MLContext();

            IEnumerable<ImageData> images = LoadImageFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            IDataView imageData = mLContext.Data.LoadFromEnumerable(images);
            IDataView shuffledData = mLContext.Data.ShuffleRows(imageData);

            var preprocessingPipeline = mLContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelAsKey")
                .Append(mLContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: assetsRelativePath,
                    inputColumnName: "ImagePath"));

            IDataView preProcessedData = preprocessingPipeline
                .Fit(shuffledData)
                .Transform(shuffledData);

            TrainTestData trainSplit = mLContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mLContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;

            ClassifySingleImage(mLContext, testSet, trainedModel);
            ClassifyImage(mLContext, testSet, trainedModel);

            var classifierOption = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
                WorkspacePath = workspaceRelativePath
            };

            var trainingPiepLine = mLContext.MulticlassClassification.Trainers.ImageClassification(classifierOption)
                .Append(mLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            ITransformer trainedModel = trainingPiepLine.Fit(trainSet);

        }

        public static void ClassifyImage(MLContext mLContext, IDataView data, ITransformer trainedModel)
        {
            IDataView predictionData = trainedModel.Transform(data);
            IEnumerable<ModelOutPut> predictions = mLContext.Data.CreateEnumerable<ModelOutPut>(predictionData, reuseRowObject: true).Take(10);
            Console.WriteLine("Classifying mutiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
            
        }

        private static void OutputPrediction(ModelOutPut prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Acutal Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        public static void ClassifySingleImage(MLContext mLContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutPut> predictionEngine = mLContext.Model.CreatePredictionEngine<ModelInput, ModelOutPut>(trainedModel);
            ModelInput image = mLContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
            ModelOutPut prediction = predictionEngine.Predict(image);
            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
            
        }

        public static IEnumerable<ImageData> LoadImageFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;
                {
                    var label = Path.GetFileName(file);
                    if (useFolderNameAsLabel)
                        label = Directory.GetParent(file).Name;
                    else
                    {
                        for (int index = 0; index < label.Length; index++)
                        {
                            if (!char.IsLetter(label[index]))
                            {
                                label = label.Substring(0, index);
                                break;
                            }
                        }
                    }

                    yield return new ImageData()
                    {
                        ImagePath = file,
                        Label = label
                    };
                }
            }
        }
    }

    class ImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    class ModelInput
    {
       
        public byte[] Image { get; set; }
        public UInt32 LableAsKey { get; set; }
        public string ImagePath { get; set; }
        public string Label { get; set; }

        
    }

    class ModelOutPut
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public string PredictedLabel { get; set; }
    }
}
