#!/bin/bash

# Loop through folds 0 to 4
for FOLD in {0..4}; do
  echo "Submitting job for fold $FOLD"
  # Pass the FOLD variable to the PBS script using the -v flag
  qsub -v FOLD=$FOLD start_single_fold.pbs
done

