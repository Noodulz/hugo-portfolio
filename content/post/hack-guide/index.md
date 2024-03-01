---
title: My Personal Study Guide For Delving into Hacking and Cybersecurity
subtitle: A level-by-level approach to learning the low-level fundamentals of computers for CTFs, exploit development, security etc.

# Summary for listings and search engines
summary: A level-by-level approach to learning the low-level fundamentals of computers for CTFs, exploit development, security etc.

# Date published
date: '2024-03-01T00:00:00Z'

# Date updated
lastmod: '2020-07-18T00:00:00Z'

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: ''
  focal_point: ''
  placement: 2
  preview_only: false

authors:
  - admin

tags:
  - Security

categories:
  - Guides
---

Lately I've spent this summer aimlessly flipping through PDFs of guides on programming and exploitation as well as diving into (what is probably my 5th time) CS50 to revisit the basics and learn the foundation needed to break into this mysterious field. After watching countless videos and reading hundreds of articles every night, though, I find myself getting distracted and off track jumping from resource to resource and book to article, barely getting anywhere.
<br/>

So after all that time spent, here I'll lay out a structured guide of resources and websites to keep me on track to where I need to be to get to a point of understanding CTFs and overall application security better. Hopefully it may reach some of you guys who may also be bitterly prone to distractions and hopping from guide to guide.
<br/>

## Level 1

* Even if you're a seasoned hacker or expert in CS, or whether you're a fledgling beginner, there's no better way to start off learning the basics than with [Harvard's CS50](https://www.edx.org/course/cs50s-introduction-to-computer-science). For the 2020 version at least, it'll get you deep into C programming, data structures and algorithms, concepts of memory and pointers (which is also great for understanding buffer overflows later on), Python, SQL, and application development. Basically the foundation of almost everything you'd learn in a 4-year Computer Science program.
* [Hacking: The Art of Exploitation](https://nostarch.com/hacking2.htm), which is also a great guide in exploring C, Assembly, networking concepts, exploitation and more for a beginner. An excellent book providing the foundation of hacking and its origins.
* [OverTheWire](https://overthewire.org/wargames/), especially Bandit and Natas through Krypton challenges down the list are useful ongoing wargames to learn more about Linux and web security and exploitation through hands-on experience and useful hints should you get stuck.
* [HackSplaining](https://www.hacksplaining.com/lessons), a visual look at common vulnerabilities in web applications.
* [Optional reading, but Code by Charles Petzold is also another good intro to how computers work and why to understand internals better for absolute beginners.](https://www.amazon.com/Code-Language-Computer-Hardware-Software/dp/0735611319)
* [MIT's The Missing Semester](https://missing.csail.mit.edu/) teaches you the tools needed to approach programming, hacking, and overall development (i.e. Github, Docker, Vim, IDEs, BASH scripting etc).
* [FreeCodeCamp](https://www.freecodecamp.org/) for learning HTML, CSS, JS, and web development through projects and certifications to go for (so you know how to break them ;))
<br/><br/>

## Level 2

* [TryHackMe's Zero to Hero Boxes Guide](https://blog.tryhackme.com/going-from-zero-to-hero/). There's a map for free members who can't or won't get the subscription, and a map for subscribed members. Personally I find this a much easier and similar alternative to HackTheBox, due to the numerous threads and hints and explanations in each of the boxes to help you understand concepts better. Though, it's best to keep practicing the concepts and exercises taught in the boxes as it can be so simple and easy to forget once you've finished a box.
* [Heath Adam's Ethical Hacking Course (aka TheCyberMentor)](https://www.udemy.com/course/practical-ethical-hacking/). Assumes you're a beginner to ethical hacking and teaches you everything you need to know about penetration testing for web and other applications. Especially useful if you're interested in pursuing the OSCP in the future.
* [The Nightmare course](https://guyinatuxedo.github.io/) by my good friend guyinatuxedo. It is a comprehensive online book on getting into binary exploitation through exploring CTF challenges and various other real life examples. And one of the more beginner friendly guides for those interested in malware analysis and reverse engineering in the future.
<br/><br/>
## Level 3

* [HackTheBox](https://www.hackthebox.eu/). If you get stuck, there's always [Ippsec's videos](https://www.youtube.com/c/ippsec/playlists). Playing through the retired boxes is an especially good place to start off in HTB.
* [Pwnable.kr](https://pwnable.kr/) for a cute and fun approach to pwning challenges and binary exploitation.
* [VulnHub](https://www.vulnhub.com/). Similar to HackTheBox in that it provides downloadable Virtual Machines to practice hacking into.
* [Microcorruption](https://microcorruption.com/login), an ongoing CTF focused on embedded security.
* For OSCP-like boxes on HTB and VulnHub, here's a [spreadsheet listing the boxes that are highly similar to the labs during the PWK course](https://docs.google.com/spreadsheets/d/1dwSMIAPIam0PuRBkCiDI88pU3yzrqqHkDtBngUHNCw8/edit#gid=1839402159) that also serves as another great way to build your skills overall.
<br/><br/>

## Level 4 (Or more for deeper understanding of computer and systems internals as well as other specific fields)

* [Nand2Tetris](https://www.nand2tetris.org/) teaches you how to build a computer and OS from the ground up. A deep dive into operating systems
* [Hasherezade's Malware Analysis/Reverse Engineering Guide](https://hshrzd.wordpress.com/how-to-start/)
* For a hands-on and lecture based approach to going through The Web Application Hacker's Handbook (if you're interested in web), look no further than [Sam Bowne's Securing Web Applications](https://samsclass.info/129S/129S_F16.shtml). I personally couldn't get through the book at first but having a video and exercises to follow along with through reading the book helped immensely in understanding the concepts better.
* [Awesome Gameboy Dev](https://project-awesome.org/gbdev/awesome-gbdev) for learning emulation, reverse engineering, and assembly and C for building and reversing all things Gameboys.
* [Cryptohack](https://cryptohack.org/), a relatively new and very astounding website which teaches you cryptography through programming exercises (highly recommend the use of Python for this). Starts off easy but gets exponentially harder as you progress.
* [Trail of Bit's Guide on Forensics](https://trailofbits.github.io/ctf/forensics/). There's not much friendly guides on learning computer forensics out there for those who are interested, but this guide is about the only one I found that provides a comprehensive overview of how to approach forensics challenges for if you encounter one in CTFs.
* [Systems Programming and Tools taught by Sanjiv Bhatia](http://www.cs.umsl.edu/\~sanjiv/classes/cs2750/), lectures and walkthroughs on programming in the Linux/Unix environment.
<br/><br/>

## Some other fun sites to practice on

* [HackThisSite!](https://hackthissite.org) in case Hellbound Hackers is being wonky
* [Enigma Group](https://www.enigmagroup.org/). Yet another web security training site but also I love the Alan Turing reference.
* [Try2Hack](http://www.try2hack.nl/)
* [picoCTF](https://picoctf.com/). High school based CTF but also fun for beginners as well. Also great in exploring what topics you may be specifically interested in.
* [HackNet](https://store.steampowered.com/app/365450/Hacknet/), just a really fun hacking simulator, one of the few accurate ones out there as well.
* [Don't forget to check for upcoming CTFs at ctftime.org!](http://ctftime.org/)
<br/>

Realistically, you could approach any of these resources in any order you prefer depending on your level of experience. This, though, provides to me at least a more structured and streamlined way of learning security concepts in a progressing manner. I hope this also helps future readers interested in also breaking into the field. This list may be edited in the future as needed.