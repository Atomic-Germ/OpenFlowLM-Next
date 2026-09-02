BUILD_DIR ?= ../../build/test/gemma_embedding
CMAKE ?= cmake

.PHONY: all configure test clean

all: configure
	$(CMAKE) --build $(BUILD_DIR) --config Release

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

test: all
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	$(BUILD_DIR)/test_gemma_embedding --model embed-gemma:300m -s 1

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)
