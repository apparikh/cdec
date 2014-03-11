#ifndef _PLRE_FF_H_
#define _PLRE_FF_H_

#include <vector>
#include <map>
#include <string>

#include "ff.h"

struct PLRENgramDetectorImpl;
class PLRENgramDetector : public FeatureFunction {
 public:
  // param = "filename.lm [-o <order>] [-U <unigram-prefix>] [-B <bigram-prefix>] [-T <trigram-prefix>] [-4 <4-gram-prefix>] [-5 <5-gram-prefix>] [-S <separator>]
  PLRENgramDetector(const std::string& param);
  ~PLRENgramDetector();
  virtual void FinalTraversalFeatures(const void* context,
                                      ::SparseVector<double>* features) const;
 protected:
  virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                     const HG::Edge& edge,
                                     const std::vector<const void*>& ant_contexts,
                                     ::SparseVector<double>* features,
                                     ::SparseVector<double>* estimated_features,
                                     void* out_context) const;
 private:
  PLRENgramDetectorImpl* pimpl_;
};

#endif
