# Injet

#### Overview

Injet is a versatile command-line program designed to perform seamless file 
injection and extraction within PNG images using the Least Significant Bit (LSB) method.


#### Usage

##### Injecting a file

To hide a file within a PNG image, employ the following command:

`injet insert some_file.txt some_image.png > result.png`

*if the image is not a PNG and/or uses a different color scheme than RGBA8, 
it will be converted automatically*

##### Extracting a file

To extract a file, use the following command:

`injet extract result.png`

The file `some_file.txt` will be created in the current directory


##### Inspecting an image

To get an information if the image contains a file inside and the maximum possible
size the image can contain, use the following:

`injet inspect image.png`
