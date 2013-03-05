/*
 * versionscheck.h
 *
 *  Created on: Mar 5, 2013
 *      Author: cgoll
 */

#ifndef _VERSIONSCHECK_H_
#define _VERSIONSCHECK_H_

#define DEAL_II_VERSION_LT(major,minor) (DEAL_II_MAJOR_VERSION * 10000 + \
    DEAL_II_MINOR_VERSION  <  (major)*10000 + (minor))

#define DEAL_II_VERSION_GTE(major,minor) (DEAL_II_MAJOR_VERSION * 10000 + \
    DEAL_II_MINOR_VERSION  >=  (major)*10000 + (minor))


#endif /* _VERSIONSCHECK_H_ */
