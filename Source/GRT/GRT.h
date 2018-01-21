#pragma once

#include "Core.h"
#include "ModuleManager.h"

#ifndef GRT_HEADER
# define GRT_HEADER
# define GRT_VERSION "0.2.5"
# define GRT_DEFAULT_NULL_CLASS_LABEL 0

// returns the filename (stripped of the system path) of the file using this
// macro
# define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, \
                                                        '/') + 1 : __FILE__)

// Fix VS lint since GRT_API is dynamically injected during runtime
# ifndef GRT_API
#  define GRT_API  __declspec(dllexport)
# endif // GRT_API

#endif  // GRT_HEADER

DECLARE_LOG_CATEGORY_EXTERN(GRTModule, Log, All);

class FGRTModule : public IModuleInterface {
public:

  /** IModuleInterface implementation */
  virtual void StartupModule() override;
  virtual void ShutdownModule() override;
};
