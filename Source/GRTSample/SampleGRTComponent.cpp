// Fill out your copyright notice in the Description page of Project Settings.
#include "SampleGRTComponent.h"


// Sets default values for this component's properties
USampleGRTComponent::USampleGRTComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void USampleGRTComponent::BeginPlay()
{
	Super::BeginPlay();

	FString obejctName = GetOwner()->GetName();
	FString info = FString::Printf(TEXT("test integrating GRT with %s"), *obejctName);
	this->GRTVersion = GRT::GRTBase::getGRTVersion();
	// Set the data label by object name
	Label = obejctName;
	print(info);

	// test
	uint32 gestureLabel = 1;
	GRT::MatrixFloat trainingSample;

	//For now we will just add 10 x 20 random walk data timeseries
	GRT::Random random;
	for (uint32 k = 0; k < 10; k++) {//For the number of classes
		gestureLabel = k + 1;

		//Get the init random walk position for this gesture
		GRT::VectorFloat startPos(ClassificationData.getNumDimensions());
		for (uint32 j = 0; j < startPos.size(); j++) {
			startPos[j] = random.getRandomNumberUniform(-1.0, 1.0);
		}

		//Generate the 20 time series
		for (uint32 x = 0; x < 20; x++) {

			//Clear any previous timeseries
			trainingSample.clear();

			//Generate the random walk
			uint32 randomWalkLength = random.getRandomNumberInt(90, 110);
			GRT::VectorFloat sample = startPos;
			for (uint32 i = 0; i < randomWalkLength; i++) {
				for (uint32 j = 0; j < startPos.size(); j++) {
					sample[j] += random.getRandomNumberUniform(-0.1, 0.1);
				}

				//Add the sample to the training sample
				trainingSample.push_back(sample);
			}

			//Add the training sample to the dataset
			ClassificationData.addSample(gestureLabel, trainingSample);
		}
	}
}


// Called every frame
void USampleGRTComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

