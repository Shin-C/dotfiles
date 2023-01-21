ssh jiac1@172.26.144.42

# super user
sudo -i

# update system
sudo apt update && sudo apt upgrade -y
sudo apt-get install libtool autoconf automake cmake libncurses5-dev g++ -y

# install go
sudo snap install go --classic 

# install nvim
sudo add-apt-repository ppa:neovim-ppa/unstable
sudo apt-get update
sudo apt-get install neovim

#install zsh
sudo apt install zsh -y

# oh my zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
sudo apt install python-pip thefuck
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k


# lf
sudo env CGO_ENABLED=0 go install -ldflags="-s -w" github.com/gokcehan/lf@latest
# add ~/go/bin to $PATH


# dofile
cd ~
git clone https://github.com/Shin-C/dotfiles.git
if touch ~/.zshrc; then
	:rm ~/.zshrc
fi
ln -s ~/dotfiles/zsh/.zshrc .zshrc
ln -s ~/dotfiles/vim/.vimrc .vimrc

mkdir ~/.config/lf
ln -s ~/dotfiles/lf/lfcd.sh ~/.config/lf/lfcd.sh
ln -s ~/dotfiles/lf/lfrc ~/.config/lf/lfrc


# install python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update 
sudo apt install python3.10 -y
sudo apt-get install python3-apt

sudo apt install python3.10-distutil
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

python3.10 -m pip install pandas numpy statsmodels matlibplot linearmodels
sudo cp ~/dotfiles/pythonstartup/startup.py /usr/lib/python3.10/startup.py



# fzf
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

#clipit
sudo apt-get install clipit
sudo apt-get install build-essential git automake xdotool autoconf intltool autopoint gtk+-3.0 xclip


fzf

pip install virtualenv
virtualenv venv_shinc --python python3.10
source nameofthevenv/bin/activate


pip install jupyter ipykernel
python -m ipykernel install --user --name venv_shinc
pip install --upgrade pip
python3.10 -m pip install pandas numpy statsmodels linearmodels matplotlib
jupyter notebook


jupyter-notebook --NotebookApp.token='' --NotebookApp.password='' --no-browser

Kobesha421|ssh -N -f -L localhost:8889:localhost:8888 jiac1@172.26.144.42

firefox http://localhost:8890
