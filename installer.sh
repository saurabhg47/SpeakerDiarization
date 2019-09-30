sudo apt-get update
sudo apt-get upgraade
sudo apt-get install ffmpeg
sudo apt-get install python3-distutils
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo pip install virtualenvwrapper
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv speaker
workon speaker
pip install -r requirements.txt