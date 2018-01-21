#pragma once

#include "../GRT.h"
#include "Observer.h"
#include "../Types/Vector.h"

namespace GRT {
template<class NotifyType>
class ObserverManager {
public:

  ObserverManager() {}

  ObserverManager(const ObserverManager& rhs) {
    *this = rhs;
  }

  ~ObserverManager() {}

  ObserverManager& operator=(const ObserverManager& rhs) {
    if (this != &rhs) {
      removeAllObservers();

      for (size_t i = 0; i < rhs.observers.size(); i++) {
        observers.push_back(rhs.observers[i]);
      }
    }
    return *this;
  }

  bool registerObserver(Observer<NotifyType>& newObserver) {
    // Check to make sure we have not registered this observer already
    const size_t numObservers = observers.size();

    for (size_t i = 0; i < numObservers; i++) {
      Observer<NotifyType> *ptr = observers[i];

      if (ptr == &newObserver) {
        return false;
      }
    }

    // If we get this far then we can register the observer
    observers.push_back(&newObserver);
    return true;
  }

  bool removeObserver(const Observer<NotifyType>& oldObserver) {
    // Find the old observer and remove it from the observers list
    const size_t numObservers = observers.size();

    for (size_t i = 0; i < numObservers; i++) {
      const Observer<NotifyType> *ptr = observers[i];

      if (ptr == &oldObserver) {
        observers.erase(observers.begin() + i);
        return true;
      }
    }
    return false;
  }

  bool removeAllObservers() {
    observers.clear();
    return true;
  }

  bool notifyObservers(const NotifyType& data) {
    // Notify all the observers
    const size_t numObservers = observers.size();

    for (size_t i = 0; i < numObservers; i++) {
      Observer<NotifyType> *ptr = observers[i];

      if (ptr != NULL) ptr->notify(data);
    }

    return true;
  }

protected:

  Vector<Observer<NotifyType> *> observers;
};
}
