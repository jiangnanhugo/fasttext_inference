/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include "model_inference.h"

#include <iostream>
#include <assert.h>
#include <algorithm>

namespace fasttext {

	Model::Model(std::shared_ptr<Matrix> wi,std::shared_ptr<Args> args,int32_t seed): hidden_(args->dim),grad_(args->dim), rng(seed), quant_(false){
		wi_ = wi;
		args_ = args;
		hsz_ = args->dim;
		negpos = 0;
		loss_ = 0.0;
		nexamples_ = 1;
		initSigmoid();
		initLog();
	}

	Model::~Model() {
		delete[] t_sigmoid;
		delete[] t_log;
	}

	void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi) {
		qwi_ = qwi;
	}

	

	


	void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
		assert(hidden.size() == hsz_);
		hidden.zero();
		for (auto it = input.cbegin(); it != input.cend(); ++it) {
			if (quant_) {
				hidden.addRow(*qwi_, *it);
			}
			else {
				hidden.addRow(*wi_, *it);
			}
		}
		hidden.mul(1.0 / input.size());
	}

	bool Model::comparePairs(const std::pair<real, int32_t> &l,
		const std::pair<real, int32_t> &r) {
		return l.first > r.first;
	}



	void Model::initTableNegatives(const std::vector<int64_t>& counts) {
		real z = 0.0;
		for (size_t i = 0; i < counts.size(); i++) {
			z += pow(counts[i], 0.5);
		}
		for (size_t i = 0; i < counts.size(); i++) {
			real c = pow(counts[i], 0.5);
			for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
				negatives.push_back(i);
			}
		}
		std::shuffle(negatives.begin(), negatives.end(), rng);
	}

	int32_t Model::getNegative(int32_t target) {
		int32_t negative;
		do {
			negative = negatives[negpos];
			negpos = (negpos + 1) % negatives.size();
		} while (target == negative);
		return negative;
	}

	void Model::buildTree(const std::vector<int64_t>& counts) {
		tree.resize(2 * osz_ - 1);
		for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
			tree[i].parent = -1;
			tree[i].left = -1;
			tree[i].right = -1;
			tree[i].count = 1e15;
			tree[i].binary = false;
		}
		for (int32_t i = 0; i < osz_; i++) {
			tree[i].count = counts[i];
		}
		int32_t leaf = osz_ - 1;
		int32_t node = osz_;
		for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
			int32_t mini[2];
			for (int32_t j = 0; j < 2; j++) {
				if (leaf >= 0 && tree[leaf].count < tree[node].count) {
					mini[j] = leaf--;
				}
				else {
					mini[j] = node++;
				}
			}
			tree[i].left = mini[0];
			tree[i].right = mini[1];
			tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
			tree[mini[0]].parent = i;
			tree[mini[1]].parent = i;
			tree[mini[1]].binary = true;
		}
		for (int32_t i = 0; i < osz_; i++) {
			std::vector<int32_t> path;
			std::vector<bool> code;
			int32_t j = i;
			while (tree[j].parent != -1) {
				path.push_back(tree[j].parent - osz_);
				code.push_back(tree[j].binary);
				j = tree[j].parent;
			}
			paths.push_back(path);
			codes.push_back(code);
		}
	}

	real Model::getLoss() const {
		return loss_ / nexamples_;
	}

	void Model::initSigmoid() {
		t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
		for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
			real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
			t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
		}
	}

	void Model::initLog() {
		t_log = new real[LOG_TABLE_SIZE + 1];
		for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
			real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
			t_log[i] = std::log(x);
		}
	}

	real Model::log(real x) const {
		if (x > 1.0) {
			return 0.0;
		}
		int i = int(x * LOG_TABLE_SIZE);
		return t_log[i];
	}

	real Model::sigmoid(real x) const {
		if (x < -MAX_SIGMOID) {
			return 0.0;
		}
		else if (x > MAX_SIGMOID) {
			return 1.0;
		}
		else {
			int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
			return t_sigmoid[i];
		}
	}

}
