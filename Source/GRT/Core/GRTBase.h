#pragma once

#include "../GRT.h"

namespace GRT {
class GRT_API GRTBase {
public:

  // GRTBase Constructor
  // id: a string representing the class ID of the inheriting type
  GRTBase(const FString& id = "");

  // GRTBase Destructor
  virtual ~GRTBase();

  // Copies the GRTBase variables from the GRTBase pointer to the instance
  // returns true if the copy was successfull, false otherwise
  bool           copyGRTBaseVariables(const GRTBase *GRTBase);

  // Gets the id of the class
  FString        getId() const;

  // Return the module version as FString
  static FString getGRTVersion();

  // Returns a pointer to the current instance
  GRTBase      * getGRTBasePointer();

  // Returns a const pointer to the current instance
  const GRTBase* getGRTBasePointer() const;

protected:

  // Store the id of the class(e.g. The name of the classfier)
  FString classId;
};
}
