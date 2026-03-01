#pragma once

#ifndef __TCPCHANNEL_H__
#define __TCPCHANNEL_H__

#ifndef CTCPCHANNEL_EXT_CLASS
#define CTCPCHANNEL_EXT_CLASS
#endif


#include "channel.h"

class CTCPCHANNEL_EXT_CLASS CTcpChannel :
	public CChannel
{
public:
	CTcpChannel(void);
	~CTcpChannel(void);
};

#endif //__TCPCHANNEL_H__