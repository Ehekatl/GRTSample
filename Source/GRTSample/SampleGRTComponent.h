// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "GRTSample.h"
#include "Components/ActorComponent.h"
#include "SampleGRTComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class GRTSAMPLE_API USampleGRTComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	USampleGRTComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:

	// indicate if GRT is load properly
	UPROPERTY(VisibleAnywhere)
		FString GRTVersion;
	UPROPERTY(VisibleAnywhere)
		FString Label;
	GRT::TimeSeriesClassificationData ClassificationData;
};
