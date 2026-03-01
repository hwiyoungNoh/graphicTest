//////////////////////////////////////////////////////////
// Copyright ©2015 Colorimetry Research, Inc. All Rights Reserved Worldwide. 	
// Version 1.26
//
// License: 
// This code is provided as a demonstration of the Remote Communication Software Development Kit.
// 
// This software is provided "as is" with no warranties of any kind.
// 
//////////////////////////////////////////////////////////
// MainFrm.h : interface of the CMainFrame class
//


#pragma once

#include <SerialChannel.h>
#include <CRColorimeter.h>
#include "DataView.h"
#include "ResultsView.h"

class CMainFrame : public CFrameWnd
{
	
public:
	CMainFrame();
protected: 
	DECLARE_DYNAMIC(CMainFrame)

// Attributes
public:

// Operations
public:

// Overrides
public:	
	virtual BOOL OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext);
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	virtual BOOL OnCmdMsg(UINT nID, int nCode, void* pExtra, AFX_CMDHANDLERINFO* pHandlerInfo);

// Implementation
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // control bar embedded members
	CStatusBar  m_wndStatusBar;
	CToolBar    m_wndToolBar;


// Generated message map functions
protected:

	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct, CCreateContext* pContext);
	afx_msg void OnSetFocus(CWnd *pOldWnd);
	afx_msg void OnCapture();
	afx_msg void OnSetup();
	afx_msg LRESULT WindowProc(UINT message, WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageMeasurementChanged(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageMeasurementDataChanged(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageDataError(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageDataSent(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageReceived(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageDebug(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageConnected(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT OnMessageDisconnected(WPARAM wParam, LPARAM lParam);
	afx_msg void OnSize(UINT nType, int cx, int cy) ;
	DECLARE_MESSAGE_MAP()

private:
	CCRColorimeter m_colorimeter;
    CSerialChannel* m_channel;
	CSplitterWnd m_mainSplitter;
	BOOL m_bInitSplitter;
	CDataView* m_pDataView;
	CResultsView* m_pResultsView;
protected:
	double* InitSampleSineWave();
};


