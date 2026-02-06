python -m venv env

source env/bin/activate

pip install -r requirements.txt

sudo apt install -y openjdk-21-jdk
sudo update-java-alternatives --set java-1.21.0-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc