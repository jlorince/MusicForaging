### startup
sudo apt-get install less
sudo apt-get install bzip2
sudo apt-get install screen
sudo apt-get install emacs

wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.0-Linux-x86_64.sh
bash Anaconda2-2.4.0-Linux-x86_64.sh

pip install graphlab-create

gsutil cp -r gs://music-foraging/LDA_vectors . 
gsutil cp gs://music-foraging/vocab_idx . 
