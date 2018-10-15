
#ifndef VERSIONSCHECK_H_
#define VERSIONSCHECK_H_

#define DEAL_II_VERSION_LT(major,minor) (DEAL_II_MAJOR_VERSION * 10000 + \
                                         DEAL_II_MINOR_VERSION  <  (major)*10000 + (minor))

#ifndef DEAL_II_VERSION_GTE
#define DEAL_II_VERSION_GTE(major,minor,small) (DEAL_II_MAJOR_VERSION * 10000 + \
                                                DEAL_II_MINOR_VERSION  >=  (major)*10000 + (minor))
#endif

#endif /* _VERSIONSCHECK_H_ */
