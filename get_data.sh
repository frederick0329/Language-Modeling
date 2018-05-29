echo "- Downloading Penn Treebank (PTB)"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p data
cd data
mkdir -p ptb
mv ../simple-examples/data/ptb.train.txt ./ptb/train.txt
mv ../simple-examples/data/ptb.test.txt ./ptb/test.txt
mv ../simple-examples/data/ptb.valid.txt ./ptb/valid.txt
cd ..
rm -r simple-examples
rm simple-examples.tgz
