# alias
alias cdd='cd /run/media/shinc/D/'
alias cdp='cd /run/media/shinc/D/PycharmProjects'

alias py='/usr/bin/python3.9'
alias ohmyzsh="mate ~/.oh-my-zsh"
alias v='nvim'
alias pych='cd /run/media/shinc/D/PycharmProjects'
alias vimrc='vim ~/.config/nvim/init.vim'
alias lfrc='v ~/.config/lf/lfrc'
alias vmap='vim ~/dotfiles/vim/maps.vim'
alias zshrc='nvim ~/.zshrc'
alias i3rc='nvim ~/.i3/config'
alias mshow='xrandr --output HDMI-3 --auto --left-of HDMI-1-1 --auto'
alias zsource='source ~/dotfiles/zsh/.zshrc'
alias zmap='nvim ~/dotfiles/zsh/alias.zsh'
alias zexport='nvim ~/dotfiles/zsh/export.zsh'
alias startup='sudo nvim /usr/bin/python3.9/startup.py'
alias tmuxrc='nvim ~/dotfiles/.tmux.conf'
alias dots='cd ~/dotfiles'
alias f='nvim $(fzf)'
alias stup='stup.sh'
# LFCD="/path/to/lfcd.sh"
# if [ -f "$LFCD" ]; then
# 	source "$LFCD"
# fi


# Use vim keys in tab complete menu:
bindkey -M menuselect 'h' vi-backward-char
bindkey -M menuselect 'k' vi-up-line-or-history
bindkey -M menuselect 'l' vi-forward-char
bindkey -M menuselect 'j' vi-down-line-or-history
bindkey -s '^o' 'lfcd\n'
