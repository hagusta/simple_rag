#!/bin/bash

if [[ -z $1 ]]; then
  COLLECTION_NAME="simple_rag"
else
  COLLECTION_NAME=$1
fi

QDRANT_HOST="localhost:6333"

curl -s "http://$QDRANT_HOST/collections/$COLLECTION_NAME" | jq '.result.points_count'
