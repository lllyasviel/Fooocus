# Variables
IMAGE_REPO ?= ghcr.io/neuro-inc
IMAGE_NAME ?= fooocus
IMAGE_TAG := latest
IMAGE_REF = $(IMAGE_REPO)/$(IMAGE_NAME):$(IMAGE_TAG)

# Build Docker images
build-image:
	@echo "Building Docker image: $(IMAGE_NAME):latest"
	docker build -t $(IMAGE_NAME):latest -f Dockerfile .
	@echo "Image built successfully"

# Push Docker images
push-image:
	@echo "Pushing Docker image $(IMAGE_REF)"
	docker tag $(IMAGE_NAME):latest $(IMAGE_REF)
	docker push $(IMAGE_REF)


# Phony targets
.PHONY: build-image push-image
