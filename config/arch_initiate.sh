 #!/bin/bash
# A sample Bash script, by Ryan
echo This is a installation guide only

sudo pacman -S yay

#update system
yes | yay -Syu
sudo pacman -mirrors --country Australia && sudo pacman -Syyu

# Wireless
# reference: https://boseji.com/posts/tp-link-archer-t9uh-on-arch/
sudo pacman -S gcc make binutils dkms

$ uname -a # chek your linux kernel version
# install the corresponding version
sudo pacman -Syyu linux-headers

# Install AUR Package
git clone https://aur.archlinux.org/rtl88xxau-aircrack-dkms-git.git
cd rtl88xxau-aircrack-dkms-git
makepkg -s
sudo pacman -U rtl88xxau-aircrack-dkms-git-*.pkg.tar.xz

# REBOOT might be required here

# utile
yes | yay -s firefox

#python
yes | yay -S python3
yes | yay -S binutils
yes | yay -S fakeroot
yes | yay -S gcc
yes | 2 | yay -S pycharm
yes | yay -S thunderbird
yes | yay -S sublime-text
yes | yay -S zoom

#zsh
yes | yay -S zsh
# change d
chsh -s $(which zsh)
yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

#install packages
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

#ie status
git clone git://github.com/tobi-wan-kenobi/bumblebee-status ~/.config/bumblebee-status

#vim
#start with Vundle
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
yes | yay -S vim
yes | yay -S neovim
yes | yay -S python-pynvim
# vim-plug
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
# change your vimrc!!!!!!!

vim +PluginInstall +qall


#plugin dependency
yes |yay -S cmake mono go npm nodejs jdk-openjdk
sudo pacman -Sy base-devel

cd ~/.vim/bundle/YouCompleteMe
python3 install.py --all



# wechat and pinyin
yay -S deepin-wine-wechat
yes | sudo pacman -Syu fcitx fcitx-googlepinyin fcitx-im fcitx-configtool
# add the follings to .xprofile
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS=@im=fcitx
# run wetchat
#/opt/apps/com.qq.weixin.deepin/files/run.sh

#Desktop
cd ~/Downloads
git clone https://github.com/vinceliuice/Tela-circle-icon-theme.git
git clone https://github.com/vinceliuice/Qogir-theme.git
./Qogir-theme/install.sh
./Tela-circle-icon-theme/install.sh

yes | sudo pacman -S kvantum-qt5

# aritim-dark


# Stata
yay -S ncurses5-compat-libs      
yay -S libpng12
# ssc install reghdfe
# ssc install ftools
# ssc install estout


# Syn time, otherwise, copilot will not work
sudo timedatectl set-ntp true


#auto mount
sudo mkdir /run/media/shinc/D
sudo fdisk -l 
# add this to /etc/fstab
# D disk
# D dirve /dev/sdb1
UUID=5690-07B3	/run/media/shinc/D	vfat	defaults,uid=shinc,gid=shinc	0	2
# old backup /dev/sda2
# UUID=1627964e-8ab6-4573-ba46-9e73e93c6f71	/run/media/shinc/backup	ext4	defaults,uid=shinc,gid=shinc 0 2

# Latex
yay -S texlive-most

#lf git file manager/// go install is a must
env CGO_ENABLED=0 go install -ldflags="-s -w" github.com/gokcehan/lf@latest
# add ~/go/bin to $PATH

# for lfimg
yes | yay -S ffmpegthumbnailer imagemagick poppler epub-thumbnailer-git wkhtmltopdf

