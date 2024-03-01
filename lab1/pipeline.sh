TRAIN_FOLDER="train"
TRAIN_FILENAME="train.csv"
TRAIN_PATH=$TRAIN_FOLDER/$TRAIN_FILENAME
TEST_FOLDER="test"
TEST_FILENAME="test.csv"
TEST_PATH=$TEST_FOLDER/$TEST_FILENAME


mkdir -p $TRAIN_FOLDER
mkdir -p $TEST_FOLDER
pip install -r requirements.txt
python3 data_creation.py
python3 model_preprocessing.py $TRAIN_PATH
python3 model_preprocessing.py $TEST_PATH
python3 model_preparation.py $TRAIN_PATH
python3 model_testing.py $TEST_PATH