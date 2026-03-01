#pragma once


#include "CRColorimeter.h"

// CFlickerSetupDlg dialog

class CFlickerSetupDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CFlickerSetupDlg)

public:
	CFlickerSetupDlg(CWnd* pParent = NULL);   // standard constructor
	virtual ~CFlickerSetupDlg();

// Dialog Data
	enum { IDD = IDD_FLICKERSETUP };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
	BOOL OnInitDialog();

public:
	void SetColorimeter(CCRColorimeter* pColorimeter);
	void EnableSettings(BOOL enable);
	void UpdateSettings(BOOL enable);

public:
	BOOL m_bMaxFreqFlickerSearch;
	int m_nFilterType;
	float m_fMaxFreqFlickerSearch;
	int m_nOrder;
	float m_fFrequency;
	float m_fBandwidth;
	CComboBox m_filterTypeComboBox;
	CSpinButtonCtrl m_maxFreqFlickerSearchSpinButtonCtrl;
	CSpinButtonCtrl m_bandwidthSpinButtonCtrl;
	CSpinButtonCtrl m_frequencySpinButtonCtrl;
	CSpinButtonCtrl m_orderSpinButtonCtrl;
	CButton m_maxFreqFlickerSearchCheckBox;
	CEdit m_maxFreqFlickerSearchEdit;
	CEdit m_orderEdit;
	CEdit m_frequencyEdit;
	CEdit m_bandwidthEdit;
	CCRColorimeter* m_pColorimeter;
};
