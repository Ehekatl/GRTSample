#include "../GRT.h"
#include "GRTBase.h"

namespace GRT {
GRTBase::GRTBase(const FString& id) : classId(id) {}

GRTBase::~GRTBase(void) {}

bool GRTBase::copyGRTBaseVariables(const GRTBase *base) {
  if (base == NULL) {
    UE_LOG(GRTModule, Error, TEXT("Fatal error, GRTBase point is null"));
    return false;
  }

  this->classId = base->classId;
  return true;
}

FString GRTBase::getGRTVersion() {
  FString version = GRT_VERSION;

  return version;
}

FString GRTBase::getId() const {
  return classId;
}

GRTBase * GRTBase::getGRTBasePointer() {
  return this;
}

const GRTBase * GRTBase::getGRTBasePointer() const {
  return this;
}
}
