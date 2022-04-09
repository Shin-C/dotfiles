let mapleader=" "

" Commenter config
let g:NERDSpaceDelims = 1
nnoremap <leader>/ :call NERDComment(0,"toggle")<CR>
vnoremap <leader>/ :call NERDComment(0,"toggle")<CR>

" Tab manager 
let g:tabman_toggle = '<leader>e'
let g:tabman_width = 35
nnoremap <leader>nt :tabnew

" Key mapping
" latex shortcut
nnoremap <Space>fr i\begin{frame}<Enter>\frametitle{}<Enter>\end{frame}<Enter><Esc>kkf}i
nnoremap <Space>bf i\textbf{}<Esc>T{i
nnoremap <Space>fn i\footnote{}<Esc>T{i
nnoremap <Space>ct bf<Space>a\cite{} <Esc>T{i
nnoremap <Space>eq i\begin{equation}<Enter>\end{equation}<Enter><Esc>2kA<ENTER>
nnoremap <Space>al i\begin{align*}<Enter>\end{align*}<Enter><++><Esc>2kA<ENTER>
nnoremap <Space>it i\begin{itemize}<Enter><Enter>\end{itemize}<Esc>1kA\item<Space>
nnoremap <Space>sec i\section{}<Esc>i
nnoremap <Space>ssec i\subsection{}<Esc>i
nnoremap <Space>sssec i\subsubsection{}<Esc>i
nnoremap <Space>vp i\vspace{2mm}<ENTER><Esc>
nnoremap <silent> <Space>ldoc :read $templates/LatexDoc.txt <CR><ESC>ggdd
nnoremap <silent> <Space>ls :read $templates/LatexSlides.txt <CR><ESC>ggdd


nnoremap <leader>ff <cmd>lua require('telescope.builtin').find_files()<cr>
nnoremap <leader>fg <cmd>lua require('telescope.builtin').live_grep()<cr>
nnoremap <leader>fb <cmd>lua require('telescope.builtin').buffers()<cr>
nnoremap <leader>fh <cmd>lua require('telescope.builtin').help_tags()<cr>

" remap save function
nmap zx :wq<ENTER>
nmap zz :q!<ENTER>

" normal map
" split and switch between tabs
" noremap <Tab>w <C-w><C-w>
" noremap <M-F10> :vsplit<Enter>
" noremap <M-F11> <C-w><C-w>:wq<Enter>
"noremap <C-a> ggVG
" nnoremap <CR> a<CR><Esc>
"Split window
nmap ss :split<Return><C-w>w
nmap sv :vsplit<Return><C-w>w" Move window
map sh <C-w>h
map sk <C-w>k
map sj <C-w>j
map sl <C-w>l" Switch tab
nmap <S-Tab> :tabprev<Return>
nmap <Tab> :tabnext<Return>

"nnoremap <leader>n :NERDTree<CR>
"nnoremap <leader>t :NERDTreeToggle<CR>
"nnoremap <leader>f :NERDTreeFind<CR>
" Visual map
vnoremap <Tab> >gv
vnoremap <S-Tab> <gv

"Exit Insert mode map
inoremap ii <ESC>

" Command line mapping
noremap <M-1> :20vs .<CR>
