% LLNCS DOCUMENT CLASS -- local compile-friendly version
% Based on the Springer LNCS layout conventions requested by the user.
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{llncs}[2024/01/29 v2.24 local compile-friendly LNCS class]

\newif\if@runhead
\@runheadfalse
\DeclareOption{runningheads}{\@runheadtrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[twoside]{article}

\RequirePackage{ifthen}
\RequirePackage{amsthm}

\setlength{\textwidth}{12.2cm}
\setlength{\textheight}{19.3cm}
\setlength{\oddsidemargin}{63pt}
\setlength{\evensidemargin}{63pt}
\setlength{\topmargin}{16pt}
\setlength{\headsep}{16pt}
\setlength{\footskip}{30pt}
\setlength{\parindent}{15pt}
\setlength{\parskip}{0pt}
\setcounter{secnumdepth}{3}

\renewcommand\small{%
  \@setfontsize\small\@ixpt{11}%
  \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
  \abovedisplayshortskip \z@ \@plus2\p@
  \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
  \belowdisplayskip \abovedisplayskip
}
\let\footnotesize\small

\def\abstractname{Abstract.}
\def\keywordname{{\bf Keywords:}}
\def\figurename{Fig.}
\def\tablename{Table}
\def\proofname{Proof}

\newcommand\subtitle[1]{\gdef\@subtitle{#1}}
\newcommand\institute[1]{\gdef\@institute{#1}}
\newcommand\titlerunning[1]{\gdef\@titlerunning{#1}}
\newcommand\authorrunning[1]{\gdef\@authorrunning{#1}}
\newcommand\inst[1]{\textsuperscript{#1}}
\newcommand\email[1]{{\ttfamily #1}}
\newcommand\orcidID[1]{\textsuperscript{[#1]}}
\newcommand\keywords[1]{\par\addvspace\baselineskip\noindent\keywordname\enspace #1}

\gdef\@subtitle{}
\gdef\@institute{}
\gdef\@titlerunning{}
\gdef\@authorrunning{}

\renewcommand\maketitle{%
  \thispagestyle{empty}%
  \begin{center}%
  {\Large\bfseries\boldmath \@title \par}%
  \ifx\@subtitle\@empty\else
    \vskip .4cm
    {\large\bfseries\boldmath \@subtitle \par}%
  \fi
  \vskip .8cm
  {\normalsize \@author \par}%
  \vskip .35cm
  {\small \@institute \par}%
  \end{center}%
  \markboth{%
    \ifx\@authorrunning\@empty\@author\else\@authorrunning\fi%
  }{%
    \ifx\@titlerunning\@empty\@title\else\@titlerunning\fi%
  }%
  \vskip .8cm
}

\renewenvironment{abstract}{%
  \list{}{\leftmargin=1cm\rightmargin=1cm}%
  \item[\hskip\labelsep\bfseries\abstractname]%
  \small
}{\endlist}

\renewcommand\section{\@startsection{section}{1}{\z@}%
  {-18\p@ \@plus -4\p@ \@minus -4\p@}%
  {12\p@ \@plus 4\p@ \@minus 4\p@}%
  {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-18\p@ \@plus -4\p@ \@minus -4\p@}%
  {8\p@ \@plus 4\p@ \@minus 4\p@}%
  {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-18\p@ \@plus -4\p@ \@minus -4\p@}%
  {-0.5em \@plus -0.22em \@minus -0.1em}%
  {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
  {-12\p@ \@plus -4\p@ \@minus -4\p@}%
  {-0.5em \@plus -0.22em \@minus -0.1em}%
  {\normalfont\normalsize\itshape}}

\renewcommand\@makecaption[2]{%
  \small
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\bfseries #1.} #2}%
  \ifdim \wd\@tempboxa >\hsize
    {\bfseries #1.} #2\par
  \else
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip
}

\def\ps@headings{%
  \let\@oddfoot\@empty
  \let\@evenfoot\@empty
  \def\@evenhead{\normalfont\small\rlap{\thepage}\hfil\leftmark}%
  \def\@oddhead{\normalfont\small\rightmark\hfil\llap{\thepage}}%
  \def\sectionmark##1{\markright{##1}}%
  \def\subsectionmark##1{}%
}
\def\ps@plain{%
  \let\@oddhead\@empty
  \let\@evenhead\@empty
  \def\@oddfoot{\hfil\thepage\hfil}%
  \let\@evenfoot\@oddfoot
}
\if@runhead
  \pagestyle{headings}
\else
  \pagestyle{plain}
\fi

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}

\endinput
