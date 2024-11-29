# Fashion Attribute Classification with EfficientNetB0
This project performs multi-label classification on fashion product images using a custom EfficientNetB0-based deep learning model. The model predicts various attributes such as category, sleeve length, and pattern for each image, aiming to improve the accuracy and efficiency of fashion attribute classification.  

## Project Overview
Type: Multi-label image classification  
Model: EfficientNetB0-based architecture  
Dataset: 70,000 training images, 30,000 test images of fashion products  
Labels: Multiple attributes such as categories and other fashion features  
Task: Predicting various attributes for each product image using image processing techniques and deep learning.  
## Dataset
The dataset consists of fashion product images labeled with various attributes. The train set contains 70,000 images, and the test set includes 30,000 images. Each image has associated labels that describe product categories and attributes.  
  
The dataset can be loaded into memory using custom generators that preprocess the images and corresponding labels in batches.
  
## Model Architecture
The core of the model is the EfficientNetB0 architecture, pretrained on ImageNet. We added custom layers to handle multi-label classification for fashion attributes:  

```python
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
outputs = [Dense(len(attribute_classes[attr]), activation='softmax', name=attr) for attr in attribute_classes]
model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
### Key points:
*EfficientNetB0* is used as the backbone for its efficient and powerful feature extraction.  
*GlobalAveragePooling2D* is used to reduce the feature maps before flattening them.  
Multiple dense layers with softmax activation are used to predict the categories and attributes.  
  
## Data Generator
A custom data generator is used to efficiently load and preprocess the images in batches during training:  
```python
def image_generator(df, batch_size=32):
    while True:
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df.iloc[start:end]
            images = []
            labels = []
            for _, row in batch_df.iterrows():
                image_id = str(row['id']).zfill(6) + '.jpg'
                img = tf.keras.preprocessing.image.load_img(os.path.join(train_image_dir, image_id), target_size=(224, 224))
                img = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img)

                labels_batch = [train_labels[attr][i] for i, attr in enumerate(attribute_classes.keys())]
                labels.append(labels_batch)

            yield np.array(images), [np.array(label) for label in zip(*labels)]
```
This generator loads and processes images in real-time to avoid memory overflow while handling large datasets. It also prepares the label data for multi-output training.  

## How to Use
### Clone the repository
```bash
git clone https://github.com/anweshaprakash/visual-taxonomy.git
cd visual-taxonomy
```
### Install dependancies
```bash
pip install -r requirements.txt
```
### Training the model
To start training the model with your dataset:
```bash
python train.py --dataset path_to_dataset --epochs 10
```
## Conclusion and Future Work
The EfficientNetB0-based model performs well in multi-label classification tasks for fashion products. Future work could include experimenting with other architectures like ResNet or MobileNet and further fine-tuning the hyperparameters for even better performance.

## Acknowledgments
The [EfficientNet Paper](https://arxiv.org/abs/1905.11946) for the model inspiration.  
Dataset provided by [Meesho](https://www.meesho.com/)

