#include "GRT.h"

#define LOCTEXT_NAMESPACE "FGRTModule"

// New log caegory for GRT module
DEFINE_LOG_CATEGORY(GRTModule);

void FGRTModule::StartupModule()
{
  UE_LOG(GRTModule, Warning, TEXT("GRT Module: Successfully loaded"));
}

void FGRTModule::ShutdownModule()
{
  UE_LOG(GRTModule, Warning, TEXT("GRT Module: Shutdown and clean up"));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FGRTModule, GRT)
