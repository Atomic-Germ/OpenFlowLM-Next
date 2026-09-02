/// \file auto_embedding_model.hpp
/// \brief AutoEmbeddingModel class
/// \author FastFlowLM Team
/// \date 2025-10-23
/// \version 0.9.24
/// \note This is a header file for the AutoEmbeddingModel class
#pragma once

#include <ctime>
#include <iomanip>
#include <sstream>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <any>
#include "typedef.hpp"
#include "device_runtime.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;


typedef enum : u8 {
    task_query = 0,
    task_document = 1,
    task_bitextmining = 2,
    task_clustering = 3,
    task_classification = 4,
    task_code_retrieval = 5,
    task_multilabel_classification = 6,
    task_sentence_similarity = 7,
    task_search_result = 8,
    task_summarization = 9,
} embedding_task_type_t;

extern std::unordered_set<std::string> embeddingModelTags;

class AutoEmbeddingModel {
protected:
	std::string model_path = "";
	bool is_model_loaded = false;
	std::string current_model = "";
	flm_rt::device* npu_device_inst = nullptr;

public:
	//************ Shared by all models *************/
	virtual ~AutoEmbeddingModel() = default;

	AutoEmbeddingModel(flm_rt::device* npu_device_inst, std::string current_model = "");
	/// \brief Get the current model
	/// \return the current model
	std::string get_current_model();

	/// \brief Show the model info
	/// \return the model info
	//************ Unique for each model *************/
	
	virtual void load_model(std::string model_path, json model_info, bool enable_preemption) {}
	virtual std::vector<float> embed(std::string& text, embedding_task_type_t task_type) = 0;
};
