/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#ifndef FASTTEXT_FASTTEXT_INFERENCE_H
#define FASTTEXT_FASTTEXT_INFERENCE_H

#define FASTTEXT_VERSION 12 /* Version 1b */
#define FASTTEXT_FILEFORMAT_MAGIC_INT32 793712314

#include <time.h>

#include <atomic>
#include <memory>
#include <set>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "qmatrix.h"
#include "model_inference.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

	class FastText {
	private:
		std::shared_ptr<Args> args_;
		std::shared_ptr<Dictionary> dict_;

		std::shared_ptr<Matrix> input_;
		//std::shared_ptr<Matrix> output_;

		std::shared_ptr<QMatrix> qinput_;
		//std::shared_ptr<QMatrix> qoutput_;

		std::shared_ptr<Model> model_;

		std::atomic<int64_t> tokenCount;
		clock_t start;
		void signModel(std::ostream&);
		bool checkModel(std::istream&);

		bool quant_;
		int32_t version;

	public:
		FastText();

		void getVector(Vector&, const std::string&) const;
		std::shared_ptr<const Dictionary> getDictionary() const;
		void saveVectors();
		void saveOutput();
		void saveModel();
		void loadModel(std::istream&);
		void loadModel(const std::string&);



		void wordVectors();
		void textVectors();
		void printWordVectors();
		void precomputeWordVectors(Matrix&);

		void loadVectors(std::string);
		int getDimension() const;
	};

}
#endif
