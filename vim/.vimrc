set nocompatible              " be iMproved, required
filetype off                  " required

syntax on

set shell=/usr/bin/zsh-5.8
call plug#begin('~/.vim/plugged')
"'Plug 'VundleVim/Vundle.vim'
"Plug 'ycm-core/YouCompleteMe'
" Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'lervag/vimtex'
Plug 'rhysd/vim-grammarous'
Plug 'preservim/nerdtree'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'chrisbra/csv.vim'
Plug 'gruvbox-community/gruvbox', { 'as': 'gruvbox' }
Plug 'junegunn/fzf'
Plug 'davidhalter/jedi-vim'
Plug 'tmsvg/pear-tree'
call plug#end()


" fold method
set foldtext=NeatFoldText()
function! NeatFoldText()
    let line = ' ' . substitute(getline(v:foldstart), '^\s*"\?\s*\|\s*"\?\s*{{' . '{\d*\s*', '', 'g') . ' '
    let lines_count = v:foldend - v:foldstart + 1
    let lines_count_text = '| ' . printf("%10s", lines_count . ' lines') . ' |'
    let foldchar = matchstr(&fillchars, 'fold:\zs.')
    let foldtextstart = strpart('+' . repeat(foldchar, v:foldlevel*2) . line, 0, (winwidth(0)*2)/3)
    let foldtextend = lines_count_text . repeat(foldchar, 8)
    let foldtextlength = strlen(substitute(foldtextstart . foldtextend, '.', 'x', 'g')) + &foldcolumn
    return foldtextstart . repeat(foldchar, winwidth(0)-foldtextlength) . foldtextend
endfunction
" load fold in startup
autocmd BufWinLeave *.* mkview 
autocmd BufWinEnter *.* silent loadview

set encoding=utf-8
let g:gruvbox_italic=1
let g:gruvbox_contrast_dark = 'hard' 
let g:airline_theme='gruvbox'
colorscheme gruvbox

" wildemenu
set wildmenu
set wildmode=longest:full,full
" search ignore capitalization
set ic
set number relativenumber
" spell check
set spell
set spelllang=en_us

" remap save function
nmap zx :wq<ENTER>
nmap zz :q!<ENTER>

" normal map
" split and switch between tabs
"noremap <Tab>w <C-w><C-w>
"noremap <M-F10> :vsplit<Enter>
"noremap <M-F11> <C-w><C-w>:wq<Enter>
"noremap <C-a> ggVG
nnoremap <CR> i<CR><Esc>

" Split window
nmap ss :split<Return><C-w>w
nmap sv :vsplit<Return><C-w>w" Move window
map sh <C-w>h
map sk <C-w>k
map sj <C-w>j
map sl <C-w>l" Switch tab
nmap <S-Tab> :tabprev<Return

nnoremap <Space>n :NERDTree<CR>
nnoremap <Space>t :NERDTreeToggle<CR>
nnoremap <Space>f :NERDTreeFind<CR>
" Visual map
vnoremap <Tab> >gv
vnoremap <S-Tab> <gv

" Insert map

inoremap ii <ESC>

