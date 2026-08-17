% Minimal svjour3 stub for error-checking only
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{svjour3}[2024/01/01 Springer svjour3 stub]
\DeclareOption{twocolumn}{}
\DeclareOption{smallcondensed}{}
\DeclareOption{iicol}{}
\DeclareOption*{}
\ProcessOptions\relax
\LoadClass[a4paper,twocolumn]{article}
\newcommand{\smartqed}{}
\newcommand{\journalname}[1]{}
\newcommand{\email}[1]{\texttt{#1}}
\newcommand{\institute}[1]{}
\newcommand{\at}{}
\renewenvironment{abstract}{\noindent\textbf{Abstract.}\quad\ignorespaces}{\par\medskip}
\newcommand{\keywords}[1]{\par\noindent\textbf{Keywords:} \def\and{;\ }#1\par}
