/// \file auto_embedding_model.hpp
/// \brief AutoEmbeddingModel class
/// \author OpenFlowLM Team
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
	oflm_rt::device* npu_device_inst = nullptr;

public:
	//************ Shared by all models *************/
	virtual ~AutoEmbeddingModel() = default;

	AutoEmbeddingModel(oflm_rt::device* npu_device_inst, std::string current_model = "");
	/// \brief Get the current model
	/// \return the current model
	std::string get_current_model();

	/// \brief Show the model info
	/// \return the model info
	//************ Unique for each model *************/
	
	virtual void load_model(std::string model_path, json model_info, bool enable_preemption) {}
	virtual std::vector<float> embed(std::string& text, embedding_task_type_t task_type) = 0;

	/// \brief The task prompt names this model declares, empty when it has none.
	///
	/// A model that HAS them cannot be embedded without choosing one. The prefix
	/// changes the vector materially -- nomic-embed-text measures cosine 0.914
	/// between the same text under search_query and search_document -- and the
	/// result is correctly shaped, correctly normed and deterministic either way,
	/// so a caller that got the wrong one has nothing to detect it with. The REST
	/// handler uses this to refuse rather than to pick.
	virtual std::vector<std::string> prompt_names() const { return {}; }

	/// Whether this backend applies a per-task prompt at all.
	///
	/// NOT the same as prompt_names() being non-empty: OpenGemma_Embedding
	/// declares no names and still prefixes per task (open_task_prefix()),
	/// while the BERT family has no task concept. Default false, so a new
	/// backend refuses a prompt it would otherwise silently drop.
	virtual bool supports_task_prompts() const { return false; }
};
