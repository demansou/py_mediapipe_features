# README

1. Open CLI
2. Navigate to root folder
3. Ensure Python3 is installed and PATH configured

## Setup Environment

`python -m venv .\.venv`

### Windows

`.\.venv\Scripts\activate.bat`

### *Nix

`source .\.venv\bin\activate`

## Install Dependencies

`pip install -r requirements.txt`

# Run Programs

## Facial Recognition

Adds a box around detected faces

`python app.py face`

## Face Mesh

Adds a stylized mesh on detected faces

`python app.py face_mesh`

## Body Detection

Adds a skeleton on detected bodies and hands

`python app.py body`

## Background Segmentation

Replaces background behind detected person(s)

`python app.py segment`