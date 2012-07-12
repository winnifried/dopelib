#!/bin/bash
#Copys the website for the OPTPDE Benchmark project to the
#server in Hamburg

USER=fmsi004

echo "Updating repository"
cd ..
svn up

cd Examples
echo "Rebuilding the PDF-Manual"
make pdf-doc &> /dev/null
cp description.pdf description_full.pdf

cd ../Webpage
if [ -d html ]
then
    cd html
else
	echo "Repository is broken: No html directory"
	exit 1
fi
echo "Copying Website with user "${USER}
scp *.html description_full.pdf ${USER}@webapp6.rrz.uni-hamburg.de:/srv/www/htdocs/dopelib/

if [ -d css ]
then
    cd css
else
	echo "Repository is broken: No css directory"
	exit 1
fi
scp *.css ${USER}@webapp6.rrz.uni-hamburg.de:/srv/www/htdocs/dopelib/css/
if [ -d images ]
then
    cd images
else
	echo "Repository is broken: No images directory"
	exit 1
fi
scp *.png *.jpg *.gif  ${USER}@webapp6.rrz.uni-hamburg.de:/srv/www/htdocs/dopelib/css/images

