/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

// this part is only used for inference.
#include "fasttext_inference.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>


namespace fasttext {

	FastText::FastText() : quant_(false) {}

	void FastText::getVector(Vector& vec, const std::string& word) const {
		const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
		vec.zero();
		for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
			if (quant_) {
				vec.addRow(*qinput_, *it);
			}
			else {
				vec.addRow(*input_, *it);
			}
		}
		if (ngrams.size() > 0) {
			vec.mul(1.0 / ngrams.size());
		}
	}

	void FastText::saveVectors() {
		std::ofstream ofs(args_->output + ".vec");
		if (!ofs.is_open()) {
			std::cerr << "Error opening file for saving vectors." << std::endl;
			exit(EXIT_FAILURE);
		}
		ofs << dict_->nwords() << " " << args_->dim << std::endl;
		Vector vec(args_->dim);
		for (int32_t i = 0; i < dict_->nwords(); i++) {
			std::string word = dict_->getWord(i);
			getVector(vec, word);
			ofs << word << " " << vec << std::endl;
		}
		ofs.close();
	}

	void FastText::saveOutput() {
		std::ofstream ofs(args_->output + ".output");
		if (!ofs.is_open()) {
			std::cerr << "Error opening file for saving vectors." << std::endl;
			exit(EXIT_FAILURE);
		}
		if (quant_) {
			std::cerr << "Option -saveOutput is not supported for quantized models."
				<< std::endl;
			return;
		}
		int32_t n = (args_->model == model_name::sup) ? dict_->nlabels()
			: dict_->nwords();
		ofs << n << " " << args_->dim << std::endl;
		Vector vec(args_->dim);
		for (int32_t i = 0; i < n; i++) {
			std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
				: dict_->getWord(i);
			vec.zero();
			//vec.addRow(*output_, i);
			ofs << word << " " << vec << std::endl;
		}
		ofs.close();
	}

	bool FastText::checkModel(std::istream& in) {
		int32_t magic;
		in.read((char*)&(magic), sizeof(int32_t));
		if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
			return false;
		}
		in.read((char*)&(version), sizeof(int32_t));
		if (version > FASTTEXT_VERSION) {
			return false;
		}
		return true;
	}

	void FastText::signModel(std::ostream& out) {
		const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
		const int32_t version = FASTTEXT_VERSION;
		out.write((char*)&(magic), sizeof(int32_t));
		out.write((char*)&(version), sizeof(int32_t));
	}

	void FastText::saveModel() {
		std::string fn(args_->output);
		if (quant_) {
			fn += ".ftz";
		}
		else {
			fn += ".bin";
		}
		std::ofstream ofs(fn, std::ofstream::binary);
		if (!ofs.is_open()) {
			std::cerr << "Model file cannot be opened for saving!" << std::endl;
			exit(EXIT_FAILURE);
		}
		std::cout << "signModel(ofs)" << std::endl;
		signModel(ofs);
		std::cout << "args_->save(ofs);" << std::endl;
		args_->save(ofs);
		std::cout << "dict_->save(ofs);" << std::endl;
		dict_->save(ofs);

		ofs.write((char*)&(quant_), sizeof(bool));
		if (quant_) {
			std::cout << "quantize input_->save(ofs);" << std::endl;
			qinput_->save(ofs);
		}
		else {
			std::cout << "input_->save(ofs);" << std::endl;
			input_->save(ofs);
		}

		ofs.close();
	}

	void FastText::loadModel(const std::string& filename) {
		std::ifstream ifs(filename, std::ifstream::binary);
		if (!ifs.is_open()) {
			std::cerr << "Model file cannot be opened for loading!" << std::endl;
			exit(EXIT_FAILURE);
		}
		if (!checkModel(ifs)) {
			std::cerr << "Model file has wrong file format!" << std::endl;
			exit(EXIT_FAILURE);
		}
		loadModel(ifs);
		ifs.close();
	}

	void FastText::loadModel(std::istream& in) {
		args_ = std::make_shared<Args>();
		dict_ = std::make_shared<Dictionary>(args_);
		input_ = std::make_shared<Matrix>();
		qinput_ = std::make_shared<QMatrix>();
		args_->load(in);
		if (version == 11 && args_->model == model_name::sup) {
			// backward compatibility: old supervised models do not use char ngrams.
			args_->maxn = 0;
		}
		dict_->load(in);

		bool quant_input;
		in.read((char*)&quant_input, sizeof(bool));
		if (quant_input) {
			quant_ = true;
			qinput_->load(in);
		}
		else {
			input_->load(in);
		}

		if (!quant_input && dict_->isPruned()) {
			std::cerr << "Invalid model file.\n"
				<< "Please download the updated model from www.fasttext.cc.\n"
				<< "See issue #332 on Github for more information.\n";
			exit(1);
		}

		model_ = std::make_shared<Model>(input_, args_, 0);
		model_->quant_ = quant_;
		model_->setQuantizePointer(qinput_);

	}

	void FastText::wordVectors() {
		std::string word;
		Vector vec(args_->dim);
		while (std::cin >> word) {
			getVector(vec, word);
			std::cout << word << " " << vec << std::endl;
		}
	}


	std::shared_ptr<const Dictionary> FastText::getDictionary() const {
		return dict_;
	}


	void FastText::printWordVectors() {
		wordVectors();
	}


	void FastText::precomputeWordVectors(Matrix& wordVectors) {
		Vector vec(args_->dim);
		wordVectors.zero();
		std::cerr << "Pre-computing word vectors...";
		for (int32_t i = 0; i < dict_->nwords(); i++) {
			std::string word = dict_->getWord(i);
			getVector(vec, word);
			real norm = vec.norm();
			if (norm > 0) {
				wordVectors.addRow(vec, i, 1.0 / norm);
			}
		}
		std::cerr << " done." << std::endl;
	}

	

	void FastText::loadVectors(std::string filename) {
		std::ifstream in(filename);
		std::vector<std::string> words;
		std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
		int64_t n, dim;
		if (!in.is_open()) {
			std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
			exit(EXIT_FAILURE);
		}
		in >> n >> dim;
		if (dim != args_->dim) {
			std::cerr << "Dimension of pretrained vectors does not match -dim option"
				<< std::endl;
			exit(EXIT_FAILURE);
		}
		mat = std::make_shared<Matrix>(n, dim);
		for (size_t i = 0; i < n; i++) {
			std::string word;
			in >> word;
			words.push_back(word);
			dict_->add(word);
			for (size_t j = 0; j < dim; j++) {
				in >> mat->data_[i * dim + j];
			}
		}
		in.close();

		dict_->threshold(1, 0);
		input_ = std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
		input_->uniform(1.0 / args_->dim);

		for (size_t i = 0; i < n; i++) {
			int32_t idx = dict_->getId(words[i]);
			if (idx < 0 || idx >= dict_->nwords()) continue;
			for (size_t j = 0; j < dim; j++) {
				input_->data_[idx * dim + j] = mat->data_[i * dim + j];
			}
		}
	}


	int FastText::getDimension() const {
		return args_->dim;
	}

}
