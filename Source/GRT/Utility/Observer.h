#pragma once

#include "../GRT.h"

namespace GRT {
template<class NotifyType>
class Observer {
public:

  Observer() {}

  virtual ~Observer() {}

  virtual void notify(const NotifyType& data) {}
};
}
